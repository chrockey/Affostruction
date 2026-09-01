# Dataset toolkit

Data preparation reuses [TRELLIS](https://github.com/microsoft/TRELLIS)'s toolkit almost verbatim. Rather than vendoring a fork, this directory ships **only what Affostruction adds**: a small patch against upstream plus the Affogato-specific scripts.

## Setup

```bash
git clone https://github.com/microsoft/TRELLIS.git /tmp/TRELLIS
cp -rn /tmp/TRELLIS/dataset_toolkits/* dataset_toolkits/     # upstream files
git apply --directory=dataset_toolkits dataset_toolkits/trellis.patch
```

The patch is against TRELLIS commit `6b0d647` and touches six files:

| File | Change |
|---|---|
| `blender_script/render.py` | EEVEE rendering path (Cycles settings guarded, denoising off so the depth pass stays crisp); view-layer name fallback for `.blend` assets |
| `render.py` | run Blender under `xvfb-run`, load `.blend` assets, EEVEE engine |
| `render_cond.py` | same, plus **`--save_depth`** — the 16-bit depth maps are Affostruction's extra conditioning signal |
| `build_metadata.py` | accept `cond_rendered` in place of `rendered`, so datasets with conditioning renders only (Affogato) still progress through voxelization / latent encoding |
| `voxelize.py` | read `mesh.ply` from `renders_cond/` when `renders/` is absent |
| `encode_ss_latent.py` | import the model registry from `affostruction` instead of `trellis` |

Everything else upstream is used as-is.

## Files added here

| File | Purpose |
|---|---|
| `datasets/Affogato.py` | Affogato metadata builder (UID → sha256, split assignment) |
| `process_affogato.py` | Affogato raw data (G-Objaverse tars + Objaverse GLBs + annotations) → TRELLIS format |
| `affogato_io.py` | mesh / depth / camera readers for the raw Affogato bundles |
| `metadata/affogato_{train,val,test}.txt` | official object splits |
| `sample_gt_points.py` | ground-truth 100k-point clouds for Chamfer distance / F-score |
| `render_gt.py` + `blender_script/render_gt.py` | ground-truth RGB + normal renders (and the camera) for PSNR / LPIPS |
| `sample_pred_points.py` | the same sampling applied to predicted meshes (`mesh.glb`) |

See [`../DATASET.md`](../DATASET.md) for the end-to-end commands.
