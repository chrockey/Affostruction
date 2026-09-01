# Evaluation manifests

Object lists for the benchmarks. `sha256` identifies a processed sample (`renders_cond/{sha256}/`, `voxels/{sha256}.ply`, ...). Everything that needs a test set reads these files — training validation, `predict.py`, `eval.py` and the ground-truth scripts in `dataset_toolkits/` — so evaluation never depends on a locally generated `metadata.csv`.

- `toys4k_test.txt` — the 1,250 Toys4k objects used for the 3D reconstruction benchmark, one sha256 per line.
- `affogato_test.txt` — the 1,000 Affogato test objects as `sha256,objaverse_uid`, where `objaverse_uid` is the original Objaverse 1.0 / Affogato UID. Each object carries five query–heatmap pairs; the complete-input protocol uses the first.
