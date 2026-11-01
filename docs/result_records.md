# Complete task measurement records

`results/06_generalization/measurements/` contains the scientific records
behind all fifteen task curves. Each case's `index.json` lists its complete
accuracy or timing files and resource exponents. The records preserve sample
identities, paired task metrics, sufficient pixel counts, multiview grouping
and the original CUDA-event or synchronized wall-clock observations.
ChangeFormer supplies its complete evaluation-set TP/FP/FN/TN counts.
Sintel records use the Final rendering and all 1,041 labeled pairs.
Pi3 accuracy uses ten complete input groups with 596 valid views in total.
Height records contain the task's pixel-error statistics.

```bash
python scripts/analyze_generalization_measurements.py --output outputs/complete_task_curves.json
```

This CPU entry reconstructs every point's task score and latency from its
declared records, then checks the published curve values. It pools pixel
counts for overlap metrics, uses the original per-image averages for image
metrics, pools squared pixel errors for height RMSE and averages per-view
point errors for multiview reconstruction. Saved FP32 per-pair EPE means
carry their corresponding rounding when pooled.

An identical execution endpoint has speedup one in the displayed curves.
Its independently observed timing is retained where measured. The GAMUS
k=17 endpoint uses its identical dense execution and task result.

## Timing statistics

Task configurations specify `timing_summary` and `speedup_summary` explicitly.
`pooled_median` takes the median across all retained calls for the declared
input. `median_weighted_round_medians` first combines each round's shape
strata with dataset-frequency weights, then takes the median over rounds.
For one shape, this is the median of that method's round medians.

| Tasks | Latency statistic | Speedup statistic |
| --- | --- | --- |
| Polyp-PVT, canopy SegFormer, ChangeFormer, GMA, GLPN | Pooled-call median | Ratio of reported dense and sparse latencies |
| HitNet, IML-ViT, TruFor | Median of shape-weighted round medians | Ratio of reported latencies |
| TUR TransWeather, Fusformer, GECO2 | Median of round medians | Ratio of reported latencies |
| AltiMap, Pi3 | Median of round medians | Median of within-round dense/sparse ratios |

The evaluation entry reports the primary statistic in `latency_ms`, the
speedup in `speedup`, and the declared estimator names. It retains every
call, round median and source-shape weight. `paired_round_speedups` supports
the paired variation shown with the curves. Pi3 uses two rounds of five
calls; the other case settings remain explicit in their configuration.
Accuracy and timing run independently, with the original preprocessing,
weights, precision and timing scopes described in the task documentation.

GAMUS RMSE pools squared pixel errors over all three test cities.
Independent timing uses the fixed `PHL_4085` input from the same 1,024-pixel
input shape. GMA k=16 and k=17 execute the dense path and therefore display
unit speedup; their independent observed timing ratios remain in the records.

## Dense-equivalent budget endpoints

When every replaced attention call uses the dense path, the reported endpoint
reuses the dense baseline and has unit speedup. `latency_source` explicitly
marks this rule; repeated independent timing is unnecessary for that endpoint.
Where separate calls were recorded, `observed_dense_median_ms` and
`observed_sparse_median_ms` preserve their medians, and `timing_file` links
the unchanged per-call arrays. These observations are separate from the reused
reporting value. AltiMap k=17 has no separate absolute latency entry; its
dense-equivalent curve endpoint is zero metric change and unit speedup.
