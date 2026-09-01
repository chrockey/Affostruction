"""
Run the reconstruction pipeline over a dataset test split and store predictions.

Runs ReconstructionPipeline on each test object and stores per-sample
predictions for the downstream metric scripts:

    {output_base}/{dataset}/{model_tag}/{num_views}/{sha256}/outputs.npz
        - sparse_structure: (N, 3) int voxel coords at 64^3 (SS decoder output)
    {.../sha256}/mesh.glb          (only with --save_mesh; used for CD/F-score)
    {.../sha256}/rgb.png, normal.png  (only with --gt_renders_dir; PSNR / LPIPS)
    {.../sha256}/views.txt         (view indices used)

Protocol:
    - objects come from manifests/{dataset}_test.txt, skipping any without
      renders_cond/{sha256}/transforms.json
    - view selection: toys4k = np.linspace over the 24 frames;
      Affogato 1-view = frame 0, k-view = frame 0 + (k-1) frames drawn with
      np.random.RandomState(int(sha256[:16], 16) % 2**32) from the rest, sorted
    - sampler: pipeline defaults from TRELLIS-image-large pipeline.json
      (FlowEulerGuidanceIntervalSampler, steps=25, cfg_strength=5.0,
      cfg_interval=[0.5, 1.0], rescale_t=3.0, sigma_min=1e-5)
    - seed = 1 (torch.manual_seed inside pipeline.run, per sample)

Usage:
    # Released checkpoint (HF), Toys4k, single view
    python predict.py --datasets toys4k --num_views 1

    # Local training output dir, Affogato
    python predict.py --ckpt outputs/stage1_reconstruction \\
        --datasets affogato --num_views 1

    # Multi-GPU sharding
    CUDA_VISIBLE_DEVICES=0 python predict.py ... --rank 0 --world_size 4
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os

os.environ.setdefault("SPCONV_ALGO", "native")

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_manifest(path: Path, dataset_root: Path, dataset_name: str) -> list:
    """Read sha256s from a manifest, keeping those with conditioning renders.

    Manifest lines are ``sha256`` or ``sha256,source_uid``; blank lines are
    ignored.
    """
    renders_dir = dataset_root / dataset_name / "renders_cond"
    with open(path) as f:
        shas = [line.split(",")[0].strip() for line in f if line.strip()]

    valid = [sha for sha in shas if (renders_dir / sha / "transforms.json").exists()]
    if len(valid) < len(shas):
        print(f"[{dataset_name}] Skipped {len(shas) - len(valid)} objects without conditioning renders")
    return valid


def select_view_indices(n_views_total, num_views, dataset_name, sha256, view_idx=None):
    """View selection: see the module docstring."""
    num_views = min(num_views, n_views_total)
    if dataset_name == "affogato":
        if num_views == 1:
            return np.array([view_idx if view_idx is not None else 0])
        seed = int(sha256[:16], 16) % (2**32)
        rng = np.random.RandomState(seed)
        remaining = rng.choice(np.arange(1, n_views_total), size=num_views - 1, replace=False)
        return np.sort(np.concatenate([[0], remaining]))
    return np.linspace(0, n_views_total - 1, num_views, dtype=int)


def load_multiview_data(renders_dir, view_indices):
    """Load selected views into the ReconstructionPipeline input_dict format."""
    with open(os.path.join(renders_dir, "transforms.json")) as f:
        metadata = json.load(f)

    images, depths, alphas, camera_params = [], [], [], []
    for view_idx in view_indices:
        frame = metadata["frames"][view_idx]
        rgb_filename = os.path.basename(frame["file_path"])
        depth_filename = rgb_filename.replace(".png", "_depth.png")

        rgb_image = Image.open(os.path.join(renders_dir, rgb_filename)).convert("RGBA")
        depth_array = (
            np.array(Image.open(os.path.join(renders_dir, depth_filename))).astype(np.float32)
            / 65535.0
        )
        alpha_array = np.array(rgb_image)[:, :, 3].astype(np.float32)

        images.append(rgb_image)
        depths.append(depth_array)
        alphas.append(alpha_array)
        camera_params.append(
            {
                "transform_matrix": frame["transform_matrix"],
                "depth_min": frame["depth"]["min"],
                "depth_max": frame["depth"]["max"],
                "camera_angle_x": frame["camera_angle_x"],
            }
        )

    return {
        "images": images,
        "depths": depths,
        "alphas": alphas,
        "camera_params": camera_params,
    }


def render_from_gt_camera(outputs, camera_path, sample_dir, resolution=512, radius=2.0, fov=40.0):
    """Render the prediction from a ground-truth benchmark camera.

    Writes ``rgb.png`` (gaussian splatting) and ``normal.png`` (mesh) next to the
    other per-object outputs; these feed the PSNR / LPIPS metrics. ``camera.txt``
    holds a single "yaw pitch" pair in radians, written by
    ``dataset_toolkits/render_gt.py``.
    """
    from PIL import Image as PILImage

    from affostruction.utils import render

    if not camera_path.exists():
        return False
    if "gaussian" not in outputs or "mesh" not in outputs:
        raise ValueError("--gt_renders_dir needs mesh + gaussian decoding (pass --save_mesh)")

    with open(camera_path) as f:
        yaw, pitch = (float(v) for v in f.readline().split())
    extrinsics, intrinsics = render.yaw_pitch_r_fov_to_extrinsics_intrinsics(
        [yaw], [pitch], [radius], [fov]
    )

    rgb = render.render_frames(
        outputs["gaussian"][0],
        extrinsics,
        intrinsics,
        options={"resolution": resolution, "bg_color": (0, 0, 0)},
        verbose=False,
    )["color"][0]
    normal = render.render_frames(
        outputs["mesh"][0], extrinsics, intrinsics, options={"resolution": resolution}, verbose=False
    )["normal"][0]
    PILImage.fromarray(rgb).save(sample_dir / "rgb.png")
    PILImage.fromarray(normal).save(sample_dir / "normal.png")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage-1 benchmark inference on test splits")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="chrockey/Affostruction",
        help="HF repo id, or a local stage-1 training "
        "output dir (config.json + ckpts/denoiser_ema*.pt)",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Subdir name for outputs (default: basename of --ckpt, e.g. Affostruction)",
    )
    parser.add_argument("--datasets", type=str, default="toys4k", help="Comma-separated list")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Object list (default: manifests/{dataset}_test.txt)",
    )
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument(
        "--view_idx", type=int, default=None, help="Custom single-view index for Affogato"
    )
    parser.add_argument("--data_root", type=str, default="trellis_data")
    parser.add_argument(
        "--output_base", type=str, default=os.path.join(REPO_ROOT, "results", "benchmark")
    )
    parser.add_argument("--seed", type=int, default=1, help="Per-sample sampling seed")
    parser.add_argument("--ss_steps", type=int, default=None, help="Override SS sampler steps")
    parser.add_argument(
        "--ss_cfg_strength", type=float, default=None, help="Override SS CFG strength"
    )
    parser.add_argument(
        "--gt_renders_dir",
        type=str,
        default=None,
        help="Benchmark GT render dir ({sha256}/camera.txt); renders the prediction from "
        "the same camera into rgb.png + normal.png for the PSNR / LPIPS metrics "
        "(implies --save_mesh)",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Also decode mesh+gaussian and bake a textured mesh.glb (needed for CD/F-score; "
        "requires nvdiffrast + diff_gaussian_rasterization)",
    )
    parser.add_argument("--force", action="store_true", help="Redo samples with existing outputs")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    if args.gt_renders_dir:
        args.save_mesh = True

    from affostruction import ReconstructionPipeline

    pipeline = ReconstructionPipeline.from_pretrained(args.ckpt).cuda()

    ss_params = {}
    if args.ss_steps is not None:
        ss_params["steps"] = args.ss_steps
    if args.ss_cfg_strength is not None:
        ss_params["cfg_strength"] = args.ss_cfg_strength

    model_tag = args.model_tag or os.path.basename(args.ckpt.rstrip("/"))
    data_root = Path(args.data_root)
    output_base = Path(args.output_base)

    for dataset_name in [d.strip() for d in args.datasets.split(",")]:
        manifest = Path(args.manifest or os.path.join(REPO_ROOT, "manifests", f"{dataset_name}_test.txt"))
        test_samples = load_manifest(manifest, data_root, dataset_name)
        print(f"[{dataset_name}] {len(test_samples)} objects from {manifest.name}")
        if args.world_size > 1:
            test_samples = test_samples[args.rank :: args.world_size]
            print(f"[shard {args.rank}/{args.world_size}] {len(test_samples)} samples")

        base_output_dir = output_base / dataset_name / model_tag / str(args.num_views)
        base_output_dir.mkdir(parents=True, exist_ok=True)

        n_done, n_skip, n_err = 0, 0, 0
        for i, sha256 in enumerate(test_samples):
            sample_dir = base_output_dir / sha256
            outputs_path = sample_dir / "outputs.npz"
            expected = [outputs_path]
            if args.save_mesh:
                expected.append(sample_dir / "mesh.glb")
            if args.gt_renders_dir:
                expected += [sample_dir / "rgb.png", sample_dir / "normal.png"]
            if all(path.exists() for path in expected) and not args.force:
                n_skip += 1
                continue
            print(f"[{i + 1}/{len(test_samples)}] {sha256[:16]}...", flush=True)
            try:
                renders_dir = data_root / dataset_name / "renders_cond" / sha256
                with open(renders_dir / "transforms.json") as f:
                    n_views_total = len(json.load(f)["frames"])
                view_indices = select_view_indices(
                    n_views_total, args.num_views, dataset_name, sha256, args.view_idx
                )
                input_dict = load_multiview_data(str(renders_dir), view_indices)

                formats = ["mesh", "gaussian"] if args.save_mesh else None
                outputs = pipeline.run(
                    input_dict,
                    seed=args.seed,
                    formats=formats,
                    sparse_structure_sampler_params=ss_params or None,
                    return_intermediates=True,
                )

                sample_dir.mkdir(parents=True, exist_ok=True)
                sparse_structure = outputs["coords"][:, 1:].cpu().numpy()
                np.savez_compressed(outputs_path, sparse_structure=sparse_structure)
                with open(sample_dir / "views.txt", "w") as f:
                    f.write("\n".join(str(v) for v in view_indices) + "\n")

                if args.gt_renders_dir:
                    render_from_gt_camera(
                        outputs,
                        Path(args.gt_renders_dir) / sha256 / "camera.txt",
                        sample_dir,
                    )

                if args.save_mesh:
                    from affostruction.utils import postprocessing

                    glb = postprocessing.to_glb(
                        outputs["gaussian"][0],
                        outputs["mesh"][0],
                        simplify=0.95,
                        fill_holes=True,
                        fill_holes_max_size=0.04,
                        texture_size=1024,
                        verbose=False,
                    )
                    glb.export(str(sample_dir / "mesh.glb"))
                n_done += 1
            except Exception as e:
                print(f"  error: {e}")
                n_err += 1

        print(f"[{dataset_name}] done={n_done} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
