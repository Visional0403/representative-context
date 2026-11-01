# Model and dataset preparation

Use the repository's fixed `weights/` and `datasets/` layout. Preparation
instructions identify the actual task weights, evaluation split, source
version, filename and final destination.

- [Task weights](weights.md): checkpoint paths, training metadata,
  third-party download sources and pure-state conversion commands.
- [Datasets](datasets.md): all seventeen dataset collections, their complete
  evaluation units, archive layouts and the fixed UAVid training tiles.
- [Dataset directory reference](../datasets/README.md): expected reader paths.
- [Inference protocols](prediction.md): task inputs, metrics and timing scopes.
- [Reproduction map](reproduction.md): numerical analysis and model entry points.

The SegFormer/UAVid and two Twins task checkpoints are available from
[GitHub Releases](https://github.com/Visional0403/representative-context/releases/tag/weights-v1). [weights.md](weights.md)
lists their download commands, training metadata and required paths.
The released task states supply both the backbone and the prediction head
for paired dense/sparse evaluation.

Fixed-version Hub assets are specified in
[configs/assets/hub_assets.json](../configs/assets/hub_assets.json):

```bash
python scripts/prepare_hub_assets.py hitnet tcd_model glpn_nyu --list
python scripts/prepare_hub_assets.py oam_tcd quickbird
python scripts/prepare_hub_assets.py fsc147
```

Architecture code is pinned in [third_party/sources.json](../third_party/sources.json)
and fetched with `scripts/prepare_sources.py`. The supplied model loaders
reuse complete task states and the original model components. Provider files
retain their source licenses and terms.
