"""Synchronized CUDA-event timing with retained per-round observations."""

import statistics
import torch

from representative_context.analysis.timing_records import call_statistics


@torch.inference_mode()
def measure(methods, warmup=10, repeats=20, rounds=3):
    for method in methods.values():
        for _ in range(warmup):
            method()
    torch.cuda.synchronize()
    names = list(methods)
    times, round_medians = {n: [] for n in names}, {n: [] for n in names}
    calls = {n: [] for n in names}
    ordinal = 0
    for index in range(rounds):
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        for method_order, name in enumerate(order):
            current = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                output = methods[name]()
                end.record()
                end.synchronize()
                current.append(start.elapsed_time(end))
                del output
            for call_id, value in enumerate(current):
                calls[name].append(dict(round_id=index, call_id=call_id, method=name,
                    method_order=method_order, measurement_order=ordinal, milliseconds=value))
                ordinal += 1
            times[name].extend(current)
            round_medians[name].append(statistics.median(current))
    return {
        name: {
            "timing_ms": times[name],
            "round_median_ms": round_medians[name],
            **call_statistics(times[name]),
            "calls": calls[name],
            "settings": dict(warmup=warmup, repeats=repeats, rounds=rounds,
                timer="cuda_event", warmup_scope="each_method_before_all_rounds",
                synchronization="after_each_call"),
        }
        for name in names
    }
