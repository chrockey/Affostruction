

# Affostruction

Official code and data for:

>[Affostruction: 3D Affordance Grounding with Generative Reconstruction](https://arxiv.org/abs/2601.09211)\
> [Chunghyun Park<sup>1</sup>](https://chrockey.github.io/),
> [Seunghyeon Lee<sup>1</sup>](https://llishyun.github.io/), and
> [Minsu Cho<sup>1,2</sup>](http://cvlab.postech.ac.kr/~mcho/)<br>
> <sup>1</sup>POSTECH and <sup>2</sup>RLWRLD<br>
> CVPR 2026, Denver.

<div align="left">
  <a href="https://arxiv.org/abs/2601.09211"><img src="https://img.shields.io/badge/arXiv-2601.09211-b31b1b.svg"/></a>
  <a href="https://chrockey.github.io/Affostruction/"><img src="https://img.shields.io/static/v1?label=project%20page&message=Affostruction&color=9cf"/></a>
  <a href="https://huggingface.co/chrockey/Affostruction"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-chrockey%2FAffostruction-yellow"/></a>
</div>

<p align="center">
  <img src="assets/teaser.gif" alt="Affostruction teaser: RGB + depth -> reconstruction + affordance" width="100%"/>
</p>
<p align="center"><em>"Reconstruct what's hidden, ground where to interact."</em></p>

## Updates

- **2026-09-01** — released the training and evaluation code: data preparation, both training stages, and the three benchmark protocols.
- **2026-05-13** — initial release: inference code, pretrained checkpoints and demos.

## Installation

Requires Python 3.10 + Linux x86_64 + CUDA 12.4.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Prefix commands with `uv run`; no venv activation needed.

## Pretrained checkpoints

Auto-downloaded from [`chrockey/Affostruction`](https://huggingface.co/chrockey/Affostruction) on first run; SLAT + mesh/gaussian decoders come from [`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large). The tables below compare the released checkpoints against the numbers reported in the paper, using the protocols in [Evaluation](#evaluation). Flow-matching sampling is stochastic, so small deviations between runs are expected.

### 3D reconstruction

Toys4k, 1,250 test objects, single RGB-D view.

| | IoU ↑ | CD ↓ | F-score ↑ | PSNR-N ↑ | LPIPS-N ↓ | PSNR ↑ | LPIPS ↓ | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| Paper | 32.67 | 0.2427 | 0.0997 | 22.64 | 0.1421 | 18.84 | 0.1922 | — |
| This repo | 36.27 | 0.2269 | 0.1105 | 23.12 | 0.1295 | 19.11 | 0.1838 | [🤗 `reconstruction`](https://huggingface.co/chrockey/Affostruction/tree/main/reconstruction) |

### Complete 3D affordance grounding

Affogato, 1,000 test objects, affordance flow on ground-truth geometry.

| | aIoU ↑ | AUC ↑ | SIM ↑ | MAE ↓ | Checkpoint |
|---|---:|---:|---:|---:|:--|
| Paper | 19.1 | 72.0 | 0.426 | 0.217 | — |
| This repo | 19.3 | 72.7 | 0.414 | 0.207 | [🤗 `affordance`](https://huggingface.co/chrockey/Affostruction/tree/main/affordance) |

### Partial 3D affordance grounding

Affogato, 1,000 test objects, affordance flow on the reconstruction from a single RGB-D view.

| | aIoU ↑ | aCD ↓ | Checkpoints |
|---|---:|---:|:--|
| Paper | 9.26 | 0.1044 | — |
| This repo | 11.56 | 0.0988 | [🤗 `reconstruction`](https://huggingface.co/chrockey/Affostruction/tree/main/reconstruction) + [🤗 `affordance`](https://huggingface.co/chrockey/Affostruction/tree/main/affordance) |

| Model | Description | # Params |
|-------|-------------|---------:|
| `reconstruction` | Multi-view RGB-D sparse-structure flow | 164.6 M |
| `affordance` | Text-conditioned affordance heatmap flow | 185.4 M |

## Usage

```python
from affostruction import AffostructionPipeline

pipeline = AffostructionPipeline.from_pretrained().cuda()
outputs = pipeline.run(input_dict, queries=["Point to the part you would sit on."])

coords = outputs["coords"]                  # (N, 4) sparse voxel coords
probs  = outputs["affordance"][0]["probs"]  # (N,) per-voxel heatmap in [0, 1], paired with coords
```

```bash
uv run python examples/affostruction.py --data_dir examples/data/sample2
```

> **Need mesh / gaussian?** Pass `formats=["mesh", "gaussian"]` — decoding is opt-in.
>
> **Reconstruction only?** Drop `queries`. See `examples/reconstruction.py` for more details.

## Data preparation

Training and evaluation run on TRELLIS-format datasets: Affogato for affordance supervision, plus 3D-FUTURE / ABO / HSSD for reconstruction and Toys4k for the reconstruction benchmark. `dataset_toolkits/` ships a patch against the upstream TRELLIS toolkit (depth-enabled conditioning renders) together with the Affogato-specific scripts.

```bash
# upstream toolkit + our delta
git clone https://github.com/microsoft/TRELLIS.git /tmp/TRELLIS
cp -rn /tmp/TRELLIS/dataset_toolkits/* dataset_toolkits/
git apply --directory=dataset_toolkits dataset_toolkits/trellis.patch

ln -s /path/to/your/datasets trellis_data   # processed datasets live here
```

Then follow [DATASET.md](DATASET.md), which gives the exact per-dataset commands and the resulting directory layout.

## Training

Prepare data first — see [DATASET.md](DATASET.md). Both stages train on 8 GPUs (batch 8/GPU), AdamW lr 1e-4, EMA 0.9999, 10% condition dropout for CFG. WandB logging is optional (set `WANDB_API_KEY`; project via `WANDB_PROJECT`).

```bash
# Stage 1 — multi-view RGBD sparse-structure flow (validates on Toys4k)
torchrun --nproc_per_node=8 --standalone train.py \
    --config configs/stage1_reconstruction.json \
    --output_dir outputs/stage1_reconstruction \
    --data_dir trellis_data/3d-future,trellis_data/abo,trellis_data/hssd,trellis_data/affogato

# Stage 2 — text-conditioned affordance heatmap flow (Affogato only)
torchrun --nproc_per_node=8 --standalone train.py \
    --config configs/stage2_affordance.json \
    --output_dir outputs/stage2_affordance \
    --data_dir trellis_data/affogato
```

Checkpoints land in `ckpts/denoiser_ema*.pt` every `i_save` steps — set `save_best_only` to keep only the best-validation snapshot — and training resumes automatically from `misc_step*.pt`. A training output dir can be evaluated directly (below).

## Evaluation

Test manifests (object lists, sha256↔UID mappings) live in [`manifests/`](manifests/). `--ckpt` takes either an HF repo id or a local training output dir.

**1. Predictions.** `predict.py` runs the reconstruction pipeline over a dataset test split (1 view = frame 0, seed 1, TRELLIS sampler defaults) and writes `outputs.npz` (sparse structure) per object — plus `mesh.glb` with `--save_mesh`, needed for Chamfer / F-score.

```bash
python predict.py --ckpt chrockey/Affostruction --datasets toys4k --num_views 1 --save_mesh \
    --gt_renders_dir results/benchmark/toys4k/gt/recon_renders
python predict.py --ckpt chrockey/Affostruction --datasets affogato   --num_views 1
```

**2. Metrics.**

```bash
# 3D reconstruction (Toys4k, 1,250 objects): volumetric IoU at 64^3
python eval.py --task recon --data_dir trellis_data/toys4k \
    --pred_dir results/benchmark/toys4k/Affostruction/1

# + Chamfer distance, F-score@0.05, PSNR and LPIPS (see DATASET.md §3 for the
# ground-truth point clouds and renders)
python eval.py --task recon --data_dir trellis_data/toys4k \
    --pred_dir results/benchmark/toys4k/Affostruction/1 \
    --mesh_metrics --gt_points_dir results/benchmark/toys4k/gt/points \
    --appearance_metrics --gt_renders_dir results/benchmark/toys4k/gt/recon_renders

# complete-input affordance grounding: aIoU / AUC / SIM / MAE on the annotation
# point clouds, affordance flow conditioned on ground-truth geometry
python eval.py --task complete --ckpt chrockey/Affostruction \
    --data_dir trellis_data/affogato --annotations_dir /path/to/affogato_raw/annotations

# partial-input affordance grounding: aIoU / aCD on the *reconstructed* voxels
python eval.py --task partial --ckpt chrockey/Affostruction \
    --data_dir trellis_data/affogato --pred_dir results/benchmark/affogato/Affostruction/1
```

Affordance sampling defaults to 50 steps, cfg 1.0, noise scale 0.1 and a zero negative condition; `--all_queries` switches the partial protocol from the first query per object to all five. `--seed` (default 0) is recorded in the result JSON.

## Acknowledgments

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft.

## Citation

```bibtex
@inproceedings{park2026affostruction,
  title={Affostruction: 3D Affordance Grounding with Generative Reconstruction},
  author={Park, Chunghyun and Lee, Seunghyeon and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
