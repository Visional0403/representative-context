"""Per-task CUDA-event or synchronized wall-clock measurements."""

import statistics
import time

import torch

from representative_context.analysis.timing_records import call_statistics, summarize_strata


@torch.inference_mode()
def measure_task(
    methods, warmup=5, repeats=20, rounds=3, timer="cuda_event", before_method=None
):
    names = list(methods)
    all_times = {name: [] for name in names}
    medians = {name: [] for name in names}
    calls = {name: [] for name in names}
    ordinal = 0
    for number in range(rounds):
        offset = number % len(names)
        order = names[offset:] + names[:offset]
        for method_order, name in enumerate(order):
            if before_method is not None:
                before_method(name)
            for _ in range(warmup):
                methods[name]()
            torch.cuda.synchronize()
            values = []
            if timer == "cuda_event":
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
                for start, end in zip(starts, ends):
                    start.record()
                    methods[name]()
                    end.record()
                torch.cuda.synchronize()
                values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
            elif timer == "wall_clock":
                for _ in range(repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = methods[name]()
                    torch.cuda.synchronize()
                    values.append((time.perf_counter() - start) * 1000)
                    del output
            else:
                raise ValueError(f"Unsupported timer: {timer}")
            for call_id, value in enumerate(values):
                calls[name].append(dict(round_id=number, call_id=call_id, method=name,
                    method_order=method_order, measurement_order=ordinal, milliseconds=value))
                ordinal += 1
            all_times[name].extend(values)
            medians[name].append(statistics.median(values))
    return {
        name: {
            "timing_ms": values,
            "round_median_ms": medians[name],
            **call_statistics(values),
            "calls": calls[name],
            "settings": dict(warmup=warmup, repeats=repeats, rounds=rounds, timer=timer,
                warmup_scope="each_method_each_round",
                synchronization=("before_each_method_and_after_event_batch"
                    if timer == "cuda_event" else "before_and_after_each_call")),
        }
        for name, values in all_times.items()
    }
