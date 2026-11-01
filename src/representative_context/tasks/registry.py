"""Resolve task protocols and their released model implementations."""

from representative_context.models.prediction import load_prediction_model
from representative_context.tasks.prediction import PredictionTask

EXTRA_CASES = ("cave", "qb", "div2k", "gopro", "gma_sintel")


def create_task(case, data_root, source=None, samples=None, timing_sample_id=None):
    if case in ("cave", "qb"):
        from representative_context.tasks.fusion import FusionTask

        return FusionTask(case, data_root)
    if case in ("div2k", "gopro"):
        from representative_context.tasks.restoration import RestorationTask

        return RestorationTask(case, data_root)
    if case == "gma_sintel":
        from representative_context.tasks.motion import MotionTask

        return MotionTask(data_root, source)
    if case == "iml":
        from representative_context.tasks.forensics import ForensicsTask

        return ForensicsTask(data_root)
    if case == "gamus":
        from representative_context.tasks.height import HeightTask

        return HeightTask(data_root, samples, timing_sample_id)
    if case == "geco2":
        from representative_context.tasks.counting import CountingTask

        return CountingTask(data_root)
    if case == "pi3":
        from representative_context.tasks.geometry import GeometryTask

        return GeometryTask(data_root, samples)
    return PredictionTask(case, data_root)


def load_task_model(
    case, weights, source=None, k=16, dense=False, dense_backend="sdpa"
):
    if case in ("iml", "gamus"):
        from representative_context.models.structured_prediction import (
            load_structured_prediction_model,
        )

        return load_structured_prediction_model(case, weights, source, k, dense)
    if case == "geco2":
        from representative_context.models.counting import load_counting_model

        return load_counting_model(weights, source, k, dense)
    if case == "pi3":
        from representative_context.models.geometry import load_geometry_model

        return load_geometry_model(weights, source, k, dense)
    if case in EXTRA_CASES:
        from representative_context.models.extra_prediction import (
            load_extra_prediction_model,
        )

        return load_extra_prediction_model(
            case, weights, source, k, dense, dense_backend
        )
    return load_prediction_model(case, weights, source, k, dense)
