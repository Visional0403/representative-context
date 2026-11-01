# Cross-task results and application readouts

`budget_curves.json` contains all published budget points for fifteen
model-dataset combinations. Each record contains the task's original
metric, its relative change and independent model timings. A positive
relative loss indicates degradation; negative values indicate improvement.

`overview.json` contains the seventeen displayed operating points,
including SegFormer/UAVid and Twins/Cityscapes. For equivalent execution
endpoints the displayed speedup is one; the independently measured timing
ratio is retained in `measured_speed_ratio`.

`application_readouts.json` contains paired dense, sparse and ground-truth
readouts for canopy cover, changed area, spectral centroid, optical-flow
magnitude, mean depth and relative depth relief. It also includes the
sample groups, resampling seed, paired confidence intervals and regression
fits. Pi3 observations are views within input groups of one TUM sequence.
Resampling uses those groups. It estimates within-sequence changes.

Original metric units are retained. Relative metric changes and speed
gains are percentages; millisecond fields contain complete model latency.
Application units are explicit in each record. The linear fits include an
intercept and report `1-SSE/SST`, with SST centered on predicted readouts.


`measurements/` supplies complete task-level paired records and adopted raw
timings. Each case index associates its files with resource exponents.
`scripts/analyze_generalization_measurements.py` recomputes all ninety budget
points. Timing estimators are specified in the task configurations and
[docs/result_records.md](../../docs/result_records.md).
