# Paper results

`04_budget/segformer_uavid.json` contains the complete k=12 through k=17
SegFormer/UAVid budget series for 1K and 2K sliding-window evaluation and
native-image evaluation. All accuracy entries cover the seventy labeled
validation images. Native timing records distinguish 2160 by 3840 and
2160 by 4096 inputs; the accuracy protocol remains the complete native
validation split.

Metric values are percentages, IoU differences are percentage points,
and timings are milliseconds. Positive loss denotes a decrease relative
to the dense model. Each record includes exact layer capacities, balanced
group counts and retained interactions. Timing samples are included when
available. Accuracy and speed are independently measured.

Newly executed checks and experiments write to `outputs/`.

`03_theory/classification/` contains five fixed UAVid windows' original
sampled classification scores. Its 40960 records are sampled pixels,
with window coordinates, source images and seeds in the manifest. The
competition and signed accuracy quantities can be independently recomputed
with `scripts/analyze_classification.py`.
