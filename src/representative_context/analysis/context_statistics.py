"""Scene-level paired summaries for context restoration and correlation controls."""

import numpy as np

from representative_context.analysis.context import ARMS, PATTERNS, metrics, task_counts


def bootstrap(array, function, seed=42):
    array = np.asarray(array)
    draws = np.random.default_rng(seed).integers(0, len(array), (3000, len(array)))
    values = np.array([function(array[indices]) for indices in draws])
    return np.quantile(values, [0.025, 0.975]).tolist()


def classification_metrics(cube):
    result = metrics(cube)
    gt, dense, current = np.indices(cube.shape)
    result["label_change_percent"] = float(
        cube[dense != current].sum() / result["pixels"] * 100
    )
    result["accuracy_percent"] = float(
        cube[gt == current].sum() / result["pixels"] * 100
    )
    return result


def restoration(directory, report):
    city = report["case"] == "twins_cityscapes"
    cubes = {pattern: {arm: [] for arm in ARMS} for pattern in PATTERNS}
    scenes = []
    for scene in report["windows"]:
        item = dict(index=scene["index"], scene=scene["scene"], layouts={})
        with np.load(directory / scene["arrays"], allow_pickle=False) as values:
            dense, gt = values["dense_prediction"], values["target"]
            for pattern in PATTERNS:
                row = scene["layouts"][pattern]
                assert row["identity_max_abs_error"] < 1e-5
                part = dict(tasks={}, within_class_error_reduction_percent=[])
                for same in row["same_input_decomposition"]:
                    first = same["layouts"]["dispersed32"]["within_class_rms"]
                    second = same["layouts"]["clustered16"]["within_class_rms"]
                    part["within_class_error_reduction_percent"].append(
                        100 * (1 - first / second)
                    )
                for arm in ARMS:
                    cube = task_counts(
                        gt, dense, values[f"{pattern}_{arm}"], report["classes"]
                    )
                    assert np.array_equal(cube, np.asarray(row["tasks"][arm]["counts"]))
                    result = classification_metrics(cube)
                    cubes[pattern][arm].append(cube)
                    part["tasks"][arm] = result
                item["layouts"][pattern] = part
        scenes.append(item)
    pooled = {}
    for pattern in PATTERNS:
        base = classification_metrics(np.sum(cubes[pattern]["sparse"], axis=0))
        pooled[pattern] = {}
        for arm in ARMS:
            combined = classification_metrics(np.sum(cubes[pattern][arm], axis=0))
            pairs = np.stack(
                (
                    [
                        classification_metrics(cube)["label_changes"]
                        for cube in cubes[pattern]["sparse"]
                    ],
                    [
                        classification_metrics(cube)["label_changes"]
                        for cube in cubes[pattern][arm]
                    ],
                ),
                axis=1,
            )

            def flip_gain(sample):
                original, current = sample.sum(0)
                return 100 * (1 - int(current) / int(original))

            observed = [flip_gain(pair[None]) for pair in pairs]
            interval = (
                [min(observed), max(observed)] if city else bootstrap(pairs, flip_gain)
            )
            pooled[pattern][arm] = {
                **combined,
                "relative_disagreement_reduction_percent": flip_gain(pairs),
                "miou_change_pp": combined["miou_percent"] - base["miou_percent"],
                "accuracy_change_pp": combined["accuracy_percent"]
                - base["accuracy_percent"],
                "improved_windows": sum(
                    row["layouts"][pattern]["tasks"][arm]["label_changes"]
                    < row["layouts"][pattern]["tasks"]["sparse"]["label_changes"]
                    for row in scenes
                ),
                "relative_disagreement_reduction_interval_percent": interval,
            }
    return dict(
        scene_count=len(scenes),
        scenes=scenes,
        pooled=pooled,
        interval_kind=(
            "observed_city_window_range" if city else "paired_sequence_bootstrap_95"
        ),
        bootstrap_repeats=0 if city else 3000,
        bootstrap_seed=None if city else 42,
    )


def correlation(directory, report):
    city = report["case"] == "twins_cityscapes"
    scenes = []
    maximum = 0.0
    for scene in report["windows"]:
        rows = []
        with np.load(directory / scene["arrays"], allow_pickle=False) as arrays:
            for block in scene["blocks"]:
                for row in block["interventions"]:
                    raw = arrays[
                        f"s{block['stage']}b{block['block']}_f{row['reassignment_fraction']}_seed{row['seed']}"
                    ]
                    assert raw.shape == (4, 16)
                    clustered, dispersed = np.sqrt(raw[1:3].sum(1) / raw[0].sum())
                    assert abs(clustered - row["aggregation_relative_l2"]["4"]) < 1e-10
                    assert abs(dispersed - row["aggregation_relative_l2"]["1"]) < 1e-10
                    assert abs(raw[3].mean() - row["neighbor_residual_cosine"]) < 1e-10
                    maximum = max(maximum, row["dense_max_relative_change"])
                    rows.append(
                        dict(
                            stage=block["stage"],
                            reassignment_fraction=row["reassignment_fraction"],
                            seed=row["seed"],
                            correlation=row["neighbor_residual_cosine"],
                            excess_error=float(clustered - dispersed),
                        )
                    )
        item = dict(scene=scene["scene"], gradient=[])
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            selected = [row for row in rows if row["reassignment_fraction"] == fraction]
            item["gradient"].append(
                dict(
                    reassignment_fraction=fraction,
                    correlation=float(
                        np.mean([row["correlation"] for row in selected])
                    ),
                    excess_error=float(
                        np.mean([row["excess_error"] for row in selected])
                    ),
                    stages=[
                        dict(
                            stage=stage,
                            correlation=float(
                                np.mean(
                                    [
                                        row["correlation"]
                                        for row in selected
                                        if row["stage"] == stage
                                    ]
                                )
                            ),
                            excess_error=float(
                                np.mean(
                                    [
                                        row["excess_error"]
                                        for row in selected
                                        if row["stage"] == stage
                                    ]
                                )
                            ),
                        )
                        for stage in range(4)
                    ],
                )
            )
        scenes.append(item)
    pooled = []
    for index, fraction in enumerate((0, 0.25, 0.5, 0.75, 1)):
        corr = np.array([row["gradient"][index]["correlation"] for row in scenes])
        gap = np.array([row["gradient"][index]["excess_error"] for row in scenes])
        pooled.append(
            dict(
                reassignment_fraction=fraction,
                correlation=float(corr.mean()),
                excess_error=float(gap.mean()),
                correlation_interval=(
                    [float(corr.min()), float(corr.max())]
                    if city
                    else bootstrap(corr, np.mean)
                ),
                excess_error_interval=(
                    [float(gap.min()), float(gap.max())]
                    if city
                    else bootstrap(gap, np.mean)
                ),
            )
        )
    return dict(
        scenes=scenes,
        pooled=pooled,
        dense_max_relative_change=maximum,
        interval_kind=(
            "observed_city_window_range" if city else "paired_sequence_bootstrap_95"
        ),
        bootstrap_repeats=0 if city else 3000,
        bootstrap_seed=None if city else 42,
    )
