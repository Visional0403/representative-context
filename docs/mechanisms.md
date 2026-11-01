# Task tolerance diagnostics

The mathematical diagnostics operate on saved scores, attention states or
regression outputs. They use FP64 CPU arithmetic for the statistical quantities.

## Classification competition

The complete sampled evidence is in `results/03_theory/classification/`.
Its manifest records five 2048-pixel UAVid windows, their source images and
coordinates, 8192 pixel positions per window, the sampling seeds and k=14.
The five compressed files contain the original dense and dispersed-layout
scores, reference labels and exact pixel indices.

Recompute the classification quantities:

```bash
python scripts/analyze_classification.py --output outputs/classification_competition.json
```

The calculation centers the class scores, projects the sparse scores onto
the dense score direction, and measures positive competitor advances against
the remaining classification margins. Tied dense winners, zero centered
dense scores and nonpositive radial scaling receive unit per-pixel bounds.
The pooled truncated bound is 3.660666843 percent, with 1.704101563 percent
observed label changes over the 40960 sampled pixels. The corresponding
counts are 314 new errors, 300 corrections and 84 wrong-class switches.
They give a sampled accuracy change of -0.034179688 percentage points.

Reconstruct these score arrays from the supplied model and original images:

```bash
python scripts/collect_classification_scores.py --output-dir outputs/classification_scores
python scripts/analyze_classification.py --input outputs/classification_scores --output outputs/reconstructed_competition.json
```

`--limit N` selects the first N declared windows for a restricted check and
records that scope. The stored coordinates and sampling positions define
the exact inputs; no new sample selection is performed after prediction.

## Spatial capacity and regression decomposition

`analysis/framework.py` also exposes the following numerical operations:

- `layout_geometry` constructs the balanced 32-region dispersed layout
  (`point32`), the 16-cluster control (`cover16`) or a local strip. It returns
  the selected sets, query-group assignments and explicit transport costs.
  These geometry checks use the stated divisible square grids and capacities.
- `prepare_attention` computes dense FP64 attention and the discrete smoothness
  bound of its probability-weighted value residual.
- `attention_diagnostics` computes retained probability mass, sparse aggregation
  error, the exact aggregation-change identity and the spatial error bound for
  the same saved Q/K/V. It retains the normalization factor in the bound.
- `regression_decomposition` separates the signed residual cross term and
  perturbation energy in the change of mean squared vector error. It also
  reports EPE changes and the contributions of improving and degrading pixels.

Regression validity follows the original finite reference-vector mask. Vector
MSE averages squared Euclidean residuals over valid pixels; it retains the sum
over coordinate channels. Classification pooling sums integer counts and
competition quantities across windows before computing rates.

## Context restoration

The paired intervention records are in `results/02_mechanisms/restoration/`.
They cover seven independent UAVid sequences for each segmentation model and
one fixed window from each of the three Cityscapes validation cities.
`measurements.json` gives the image IDs, exact window coordinates and scope.
The compressed arrays contain reference labels, dense predictions and all
five intervention predictions for both dispersed and clustered layouts.

At k=14, the interventions replace the aggregation of the final two global
attention blocks while keeping their current inputs. The class partition is
defined by the frozen dense predictions. Restoring class mass changes the
attention probability assigned to each class; restoring class content changes
the conditional value means within those classes. The norm-matched condition
rescales the content-restored perturbation to the Frobenius norm of the
original sparse perturbation over the complete projected attention branch.
Complete aggregation restoration evaluates dense aggregation at these two
blocks' current inputs. The full-model dense prediction is saved separately.

```bash
python scripts/mechanism_restoration.py --case segformer_uavid --output-dir outputs/restoration_segformer
python scripts/analyze_context.py --task restoration --case segformer_uavid --output outputs/restoration_statistics.json
```

Use `--case twins_uavid` or `--case twins_cityscapes` for the other models.
`--limit N` restricts reconstruction to the first N declared windows and records
that scope. `analyze_context.py --input-dir ...` analyzes reconstructed records.
Integer GT/dense/current prediction counts are pooled before computing label
changes, new errors, corrections, accuracy and class IoU.

## Spatial-correlation intervention

`results/02_mechanisms/correlation/` supplies the corresponding frozen
attention measurements. Each global attention block uses sixteen fixed
queries. Paired K/V reassignment varies from zero to complete reassignment
within each quadrant, preserving dense aggregation, the local neighborhood
and global anchors. The dispersed and clustered contexts have identical
capacity, shifted 3-by-3 local neighborhoods, global anchors and remote
quadrant quotas. Only the organization of their remaining remote keys differs.

```bash
python scripts/inspect_context_layout.py --kv-side 64 --query-side 512 --capacity 64 --cluster 4 --output outputs/context_layout.json
python scripts/mechanism_correlation.py --case twins_uavid --output-dir outputs/correlation_twins
python scripts/analyze_context.py --task correlation --case twins_uavid --output outputs/correlation_statistics.json
```

The CPU layout inspection checks the invariants for every query position.
Saved arrays retain each query's dense aggregation energy, clustered and
dispersed error energies, and neighboring residual similarity. Statistics first average reassignment seeds within each operation, then
operations within a window. The four Twins stages contain 1, 1, 5 and 2
global operations and therefore contribute in those proportions. Stagewise
summaries average operations within each stage. The separate spatial-capacity
diagnostic samples one operation per stage and uses equal stage weights.
UAVid intervals use 3000 paired resamples of the seven independent sequences
with seed 42. Cityscapes summaries report the observed range of its three
fixed city windows. Resampling keeps each window's paired measurements
together. The numerical dense-invariance checks accompany the measurements.

## Cross-task spatial correlation

The complete mechanism-generalization evidence is in
`results/02_mechanisms/cross_task/`. The fixed input lists comprise:

- GMA: the median labeled Final-pass pair in each of the 23 Sintel scenes,
  measured at recurrent aggregation updates 6 and 12.
- Fusformer: all eleven CAVE test samples, measured at attention blocks
  1, 3 and 6.
- IML-ViT: the median image in each of 24 evenly spaced source directories
  from the sorted IMD2020 directory list, measured at global blocks
  3, 6, 9 and 12.

Each input uses sixteen fixed spatial queries and the same nine fraction/seed
conditions: one unchanged condition and two seeds at each of 25, 50, 75 and
100 percent reassignment. The retained capacities are 1760, 65536 and 1024
keys per query, respectively. These match the actual group sizes for the
declared task settings. Dispersed and clustered contexts retain identical
local neighbors, global anchors, quadrant quotas and padding candidates.
Only valid remote locations are reassigned; padding stays fixed.

The intervention reassigns complete score columns together with their values.
This also retains IML-ViT's learned relative-position contribution in each
moved score column. Every condition checks dense-aggregation invariance.
The task model follows its dense trajectory; diagnostic aggregations do not
replace model outputs. These mechanism runs use FP32 and disable TF32 and
GMA autocast. The separate task-evaluation configurations define the inference
precision used for the task metrics and latency curves.

```bash
python scripts/mechanism_cross_task.py --case gma --output-dir outputs/mechanism_gma
python scripts/analyze_cross_task.py --case gma --output outputs/mechanism_gma_statistics.json
```

Use `--case fusformer` or `--case iml` for the other models. `--limit N`
reconstructs the first N declared inputs for a restricted check. Complete
statistical analysis uses the full declared set; `--input-dir ...` selects
reconstructed records. The saved arrays contain the sampled dense and sparse
aggregation outputs and the neighbor-correlation sums and counts, allowing
the complete evidence to be recomputed on CPU.

Statistics average reassignment seeds within each block, then blocks within
each input, and finally independent scenes or source directories with equal
weight. Paired 95 percent intervals use 10000 source-group resamples with
seed 20261025. The saved dense-trajectory checks, numerical invariance tests
and one independent FP64 spot reconstruction per model accompany the data.

## Spatial representation across retained capacities

`results/03_theory/spatial/` supplies all four stages' sampled Q and complete
K/V tensors for seven independent UAVid windows per segmentation model.
Each block has sixteen fixed queries. Capacities are 64, 128, 256, 512,
1024 and 2048 keys on a 64-by-64 KV grid, with a full-capacity endpoint.
The inputs and coordinates are fixed in each case's manifest.

```bash
python scripts/collect_theory_evidence.py --case segformer_uavid --task spatial --output-dir outputs/spatial_qkv
python scripts/analyze_spatial_capacity.py --case segformer_uavid --check-saved --output-dir outputs/spatial_capacity
```

Use `--input-dir outputs/spatial_qkv` to analyze reconstructed tensors.
`--limit N` selects the first N declared windows. `--check-saved` compares
recomputed per-query quantities with the supplied diagnostic arrays and is
used with the published data. The full CPU analysis reconstructs 32256
head/query/layout/capacity comparisons for SegFormer and 60480 for Twins.

The geometry calculation uses normalized cell-center coordinates and an
explicit equal-mass transport plan. It records the plan's mean transport
cost and the normalization factor D_S for the probability-weighted residual
bound. The dispersed 32-region construction uses the stated divisible
dyadic grid. Its constructive upper constant is sqrt(5/2). The geometric
strip control keeps the left vertical strip of the grid; its transport
distance is (1-b/K)/2. This mathematical strip comparison has its own
definition, separate from the task's query-dependent local-window layout.
The saved quantities include actual aggregation error, retained mass,
transport and smoothness bounds, identity residuals and dense endpoints.

## Classification competition across budgets

`results/03_theory/budget_competition/` contains five SegFormer windows and
seven Twins windows at k=14, 15 and 16. Every window supplies the same 8192
pixel IDs across budgets, paired class scores, labels and full-window
competition sufficient statistics. Dense/current prediction counts or the
complete predictions support independent full-window task-count checks.

```bash
python scripts/collect_theory_evidence.py --case twins_uavid --task competition --output-dir outputs/budget_scores
python scripts/analyze_budget_competition.py --case twins_uavid --output outputs/budget_competition.json
```

The collector uses the published pixel IDs directly, evaluates the declared
windows and recomputes full-window quantities in 32768-pixel chunks. It
saves complete predictions alongside the sampled scores. The CPU analyzer
recomputes sampled competition from scores, checks signed prediction
transitions against GT and pools full-window sufficient statistics separately
from sampled pixels. `--input-dir ...` analyzes reconstructed records. These
windows retain their declared scope; complete validation-set metrics reside
in the task evaluation results.

## Signed regression changes

`results/03_theory/regression/gma_sintel/` supplies reference flow, dense flow
and ten intervention outputs for each of four complete Sintel image pairs.
The five interventions per layout comprise the sparse output, motion-content
restoration, learned-matching restoration, norm-matched matching restoration
and complete same-input aggregation restoration.

```bash
python scripts/analyze_regression.py --output outputs/regression_decomposition.json
```

This CPU calculation reconstructs all forty signed decompositions. It uses
finite GT vectors whose coordinate magnitudes are below 1e9. Vector MSE
averages squared Euclidean residuals over valid pixels, preserving the sum
over both flow coordinates. The residual cross term and perturbation energy
sum to the signed loss change. EPE changes and the contributions of improved
and degraded pixels use the same valid mask and paired outputs.

## Recurrent motion-content restoration

The GMA/Sintel mechanism uses four fixed Final image pairs. Sorted scene
indices 0, 7, 14 and 22 define alley_1, bamboo_1, market_5 and temple_3;
each scene contributes its middle adjacent pair, starting at frame 25.
The native 436-by-1024 frames are padded to a 55-by-128 attention grid.
At k=14, capacity limit 2048 yields four balanced groups of 1760 keys.
The two layouts are dispersed 32-region sampling and contiguous two-by-two
tiles within sixteen regions, balanced on the actual rectangular grid.

```bash
python scripts/mechanism_motion_restoration.py --output-dir outputs/motion_context
python scripts/analyze_motion_restoration.py --output outputs/motion_context_statistics.json
```

The collector runs the released GMA model in FP32 with autocast and TF32
disabled. It captures the original twelve-update dense trajectory and each
sparse trajectory. Motion inputs and actual aggregation outputs are recorded
at updates six and twelve. The explicit two-channel flow contribution is
projected with its matching slice of the value weights; the learned feature
contribution is the actual full value projection minus that contribution.
Both candidate layouts are applied to the same intermediate inputs when
comparing component errors.

Five interventions replay updates eleven and twelve from each sparse
trajectory's captured hidden state and coordinates. They retain sparse
aggregation, restore the explicit motion component, restore the learned
component, restore that component with matched total residual norm, or
restore complete same-input dense aggregation. Correlation lookup,
recurrent updates and convex upsampling execute normally during replay.
The identity replay is checked against the original complete sparse flow.

The norm-matched intervention rescales its complete aggregation residual
relative to the same-input dense output to the original sparse residual's
Frobenius norm. This preserves perturbation magnitude while changing the
information component. The preserved arrays include the QK input, both
motion inputs and actual outputs, replay state and per-query squared
component errors for each layout. They supplement the complete paired flow
fields in `results/03_theory/regression/gma_sintel/`.

The CPU analyzer independently checks all forty intervention outputs and
128 component RMS values. Each complete labeled image pair is the task
statistical unit; valid flow vectors have finite coordinates below 1e9 in
absolute value. Pixel-weighted pooling uses the same valid mask. Newly
collected records can be analyzed with `--input-dir outputs/motion_context`.
The collector's `--limit N` selects the first N declared pairs.

## Complete native-image difficult decisions and rare classes

`results/02_mechanisms/long_tail/segformer_uavid/` supplies all seventy native
UAVid images' paired counts at k=14 and k=16. The count axes are image,
dense-probability-margin bin, ground-truth class, dense prediction and sparse
prediction. The five fixed bin boundaries are 0, .05, .10, .20, .50 and 1;
internal-boundary ties enter the higher bin. These counts retain new errors,
corrections and switches between incorrect classes separately. All eight
classes and false positives contribute to task metrics.

```bash
python scripts/collect_long_tail.py --case segformer_uavid --output-dir outputs/uavid_difficulty
python scripts/collect_long_tail.py --case gma_sintel --output-dir outputs/flow_difficulty
python scripts/analyze_long_tail.py --output outputs/difficulty_statistics.json
```

This native UAVid collector uses the declared CPU FP32 arithmetic: divide
RGB by 255, subtract the FP32 mean and divide by the FP32 standard deviation.
Unknown label colors receive ignore ID 255. Each complete image is evaluated
once per dense or sparse setting. Free cache blocks are released after each
output reaches CPU; parameters and layout indices remain resident.

The supplied rare-class arrays contain all 1,107,672 GT-human pixels, their
dense human-versus-best-competitor probability margins and all three paired
predictions. Per-image offsets preserve their source units. Quintiles are
fixed by dense margins, with equal margins kept together. The initially
correct analysis restricts these pixels to dense predictions of Human.
The image-difficulty groups rank the 58 images containing GT humans once
by dense human IoU, with index tie-breaking. Their hardest 20%, middle 40%
and easiest 40% contain 12, 23 and 23 images. All seventy images contribute
to the full result. Intervals use paired sequence resampling, 10,000 draws
and seed 20261006. For classification accuracy, sequences follow their first
occurrence in the fixed image list. Each budget and margin bin starts a
NumPy `default_rng(20261006)` generator, resamples seven sequences with
replacement and recomputes the pooled accuracy change. The interval uses
the 2.5th and 97.5th percentiles with linear interpolation.

## Complete difficult-region optical-flow changes

The GMA records cover all 1,041 Sintel Final pairs and 23 scenes at k=14.
The model retains its native AMP setting and twelve recurrent updates;
FP32 output flows are compared with GT using CPU float64 endpoint errors.
Every pair supplies sufficient statistics for all valid pixels, its highest
dense-error quintile, official occlusion and non-occlusion masks, and GT
motion below 10, from 10 to below 40, or at least 40 pixels. Dense-error
threshold ties enter the difficult group. The hardest 20% of pairs are fixed
by descending dense EPE with scene/frame tie-breaking.

Increase and decrease magnitudes include every valid output change.
Directional counts additionally use a fixed 0.001-pixel tolerance. A bad
pixel exceeds both three pixels and five percent of GT motion. New bad
pixels and corrected bad pixels are counted separately. Confidence
intervals use paired scene resampling, 10,000 draws and seed 20261007.
Five fixed pairs retain complete GT, dense flow, sparse flow and occlusion
arrays for independent reconstruction of every region's statistics.

The CPU analyzer checks these thirty-five saved pair/region combinations
and recomputes the complete classification, rare-class and flow summaries.
The model collector's `--limit N` selects the first N sorted evaluation
units and records a restricted split. It performs no speed measurement.

## Complete-window classification certificate

`results/03_theory/full_window_certificates.json` supplies the five complete
window counts and the strict class-margin predicate. The denominator is
20,971,520 valid pixels, distinct from the sampled-pixel competition analysis.
Pool the per-window numerators and denominators with:

```bash
python scripts/analyze_full_window_certificates.py --output outputs/full_window_certificates.json
```

Recompute the complete-window certificates with:

```bash
python scripts/collect_full_window_certificates.py --save-fields --output-dir outputs/window_certificates
python scripts/analyze_full_window_certificates.py --input outputs/window_certificates/manifest.json --fields-dir outputs/window_certificates --output outputs/window_certificate_check.json
```
