# Dataset preparation

Affostruction trains on TRELLIS-format datasets plus the Affogato affordance dataset. All commands run from the repo root with `uv run`; processed datasets live under a single root (we symlink it as `trellis_data/`):

```bash
ln -s /path/to/your/datasets/trellis trellis_data
```

| Dataset | Used for | Pipeline |
|---|---|---|
| 3D-FUTURE, ABO, HSSD | stage-1 training | TRELLIS-style (§1) |
| Toys4k | stage-1 validation + reconstruction benchmark | TRELLIS-style (§1) |
| Affogato | stage-1 training, stage-2 training, affordance benchmarks | Affogato pipeline (§2) |

The final stage-1 training command consumes `trellis_data/{3d-future,abo,hssd,affo}`; stage-2 consumes `trellis_data/affogato` only.

## 1. TRELLIS-style datasets (3D-FUTURE / ABO / HSSD / Toys4k)

Same steps as upstream [TRELLIS-500K](https://github.com/microsoft/TRELLIS), with one delta: **conditioning renders include 16-bit depth maps** (`render_cond.py --save_depth`, always on) because our stage-1 model is depth-conditioned. Run `build_metadata.py` again after every step to fold the processing records into `metadata.csv`.

```bash
D=Toys4k                     # or 3D-FUTURE / ABO / HSSD
OUT=trellis_data/toys4k      # or 3d-future / abo / hssd

uv run python dataset_toolkits/build_metadata.py $D --output_dir $OUT
uv run python dataset_toolkits/download.py $D --output_dir $OUT     # Toys4k: place toys4k_blend_files.zip under $OUT/raw first
uv run python dataset_toolkits/render.py $D --output_dir $OUT       # 150 views, saves mesh.ply
uv run python dataset_toolkits/build_metadata.py $D --output_dir $OUT
uv run python dataset_toolkits/voxelize.py $D --output_dir $OUT     # 64^3 occupancy -> voxels/{sha256}.ply
uv run python dataset_toolkits/build_metadata.py $D --output_dir $OUT
uv run python dataset_toolkits/encode_ss_latent.py --output_dir $OUT  # TRELLIS ss_enc_conv3d_16l8_fp16 -> ss_latents/
uv run python dataset_toolkits/build_metadata.py $D --output_dir $OUT
uv run python dataset_toolkits/render_cond.py $D --output_dir $OUT  # 24 RGBD views @1024 -> renders_cond/ (RGBA + 16-bit depth + transforms.json)
uv run python dataset_toolkits/build_metadata.py $D --output_dir $OUT

```

Depth PNG encoding: per-view `[depth_min, depth_max] -> [0, 65535]` with background = 65535; `depth.{min,max}` are written into `transforms.json` per frame (range `radius ± √3/2`).

## 2. Affogato

Prerequisite: the ObjaverseXL metadata produced by the TRELLIS toolkit (`trellis_data/objaversexl_{sketchfab,github}/metadata.csv`). Affogato objects are keyed by Objaverse UID, and those tables provide the UID → sha256 mapping plus the aesthetic scores used to keep TRELLIS's own training objects out of the val/test splits.

Raw inputs (`--affogato_base_dir`), from the [Affogato release](https://huggingface.co/datasets/project-affogato/affogato) and its sources:

```
{affogato_base_dir}/
├── gobjaverse_reduced/{group}/{uid}.tar.gz   # 40-view G-Objaverse renders: {i:05d}.png (RGBA 512), {i:05d}_nd.exr (normal+depth), {i:05d}.json (camera)
├── hf-objaverse-v1/glbs/{xxx-yyy}/{uid}.glb  # Objaverse 1.0 meshes
└── annotations/{uid}/xyzc.npy + queries.json # 16384 pts x (XYZ + 5 heatmaps), 5 text queries
```

```bash
OUT=trellis_data/affogato

# 1) tars + glbs + annotations -> TRELLIS format.
#    Writes renders_cond/ (40 RGBD views, Unity->world rotated cameras, incl. mesh.ply),
#    affordances/{sha256}/affordance.npz ({coords: (N,3) int in [0,64), heatmap: (N,5) in [0,1],
#    query: 5 strings} — the annotation-aligned voxels used by stage 2), and metadata.csv.
uv run python dataset_toolkits/process_affogato.py Affogato --affogato_base_dir /path/to/affogato_raw --output_dir $OUT --max_workers 16
uv run python dataset_toolkits/build_metadata.py Affogato --output_dir $OUT --affogato_base_dir /path/to/affogato_raw --from_file --field cond_rendered

# 2) voxelize mesh.ply -> voxels/{sha256}.ply (stage-1 targets; standard TRELLIS voxelization —
#    the 16384 annotated surface points alone would leave holes, so stage 1 uses the mesh)
uv run python dataset_toolkits/voxelize.py Affogato --output_dir $OUT

# 3) encode voxels -> ss_latents/ss_enc_conv3d_16l8_fp16/{sha256}.npz
uv run python dataset_toolkits/encode_ss_latent.py --output_dir $OUT
uv run python dataset_toolkits/build_metadata.py Affogato --output_dir $OUT --affogato_base_dir /path/to/affogato_raw --from_file --field voxelized
```

Splits: `dataset_toolkits/metadata/affogato_{train,val,test}.txt` (train 132,393 / val 14,711 / test 1,000 UIDs). `datasets/Affogato.py` moves any val/test UID that appears in TRELLIS's own training pool into train. Checked-in test manifests (incl. the sha256↔UID mapping) live in `manifests/`.

Stage-2 training/eval data comes entirely out of step 1 (`affordances/{sha256}/affordance.npz` holds both the annotation-aligned voxel coords and the heatmaps); stages 1 and 2 therefore never share voxel dirs.

## 3. Reconstruction-benchmark ground truth (Toys4k)

The volumetric-IoU metric only needs `voxels/{sha256}.ply` from §1.

Chamfer distance / F-score compare 100k-point clouds:

```bash
# GT clouds from 100 Hammersley sphere views (Blender depth unprojection)
uv run python dataset_toolkits/sample_gt_points.py --dataset toys4k
# Predicted clouds from mesh.glb, same 100-view procedure
uv run python dataset_toolkits/sample_pred_points.py --dataset toys4k --model <model_tag> --num-views 1
```

PSNR / LPIPS compare renders from one random camera per object:

```bash
# GT rgb.png + normal.png + camera.txt per object
uv run python dataset_toolkits/render_gt.py --data_root trellis_data/toys4k \
    --output_dir results/benchmark/toys4k/gt/recon_renders
```

`predict.py --gt_renders_dir results/benchmark/toys4k/gt/recon_renders` then renders each prediction from the same camera.

`metadata.csv` is generated by these steps — it is not distributed. Training splits for Affogato ship as `dataset_toolkits/metadata/affogato_{train,val,test}.txt` and are applied by `build_metadata.py`; every evaluation split is pinned by the object lists in `manifests/`, which training validation, `predict.py`, `eval.py` and the benchmark scripts all read directly.

## metadata.csv column glossary

| Column | Meaning |
|---|---|
| `sha256` | sample id (= SHA-256 of the source UID) |
| `file_identifier` | original source id (Affogato/Objaverse UID, file path, ...) |
| `split` | `train` / `val` / `test`; empty rows are ignored by the loaders |
| `rendered`, `cond_rendered` | `renders/` (150-view) and `renders_cond/` (RGBD) done |
| `voxelized`, `num_voxels` | `voxels/{sha256}.ply` done |
| `ss_latent_ss_enc_conv3d_16l8_fp16` | `ss_latents/` encoded |
| `aesthetic_score` | ObjaverseXL aesthetic score (stage-1 filters ≥ 4.5; skipped for Affogato) |

## Finished layout (Affogato example)

```
trellis_data/affogato/
├── metadata.csv
├── renders_cond/{sha256}/{000..039}.png + {000..039}_depth.png + transforms.json + mesh.ply
├── affordances/{sha256}/affordance.npz    # coords (N,3) int32 + heatmap (N,5) float32 [0,1] + query (5 strings)
├── voxels/{sha256}.ply                    # mesh voxelization (stage-1 targets)
└── ss_latents/ss_enc_conv3d_16l8_fp16/{sha256}.npz   # mean (8,16,16,16)
```
