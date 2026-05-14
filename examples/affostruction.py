"""
Affostruction example: reconstruction + text-conditioned affordance heatmap.

Writes a colored point cloud (.ply) keyed by per-voxel heatmap probability
so the result is inspectable in MeshLab / Open3D, plus a sidecar .npz with
raw coords + probs.

Usage:
    uv run python examples/affostruction.py \
        --data_dir examples/data/sample2 --query grasp
"""

import os

os.environ.setdefault("SPCONV_ALGO", "native")

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from pathlib import Path

import fire
import numpy as np
from PIL import Image

from affostruction import AffostructionPipeline


# Real queries from the test split of the Affogato affordance dataset for
# the bundled sample objects. Keyed by sample directory basename. The first
# entry is used as the default query when ``--query`` is not passed.
SAMPLE_QUERIES = {
    "sample1": [
        "Point to the part you would sit on.",
        "Point to the armrest you would lean on.",
        "Point to the part you would place your feet.",
        "Point to the backrest you would rest your head against.",
        "Point to the base you would put your legs on.",
    ],
    "sample2": [
        "Point to the part you would sit on.",
        "Point to the part you would lean against.",
        "Point to the part you would use to support your arms.",
        "Point to the part you would use to carry the chair.",
        "Point to the part that provides stability.",
    ],
}


def load_rgbd_data(data_dir: str, num_views: int = 3) -> dict:
    """Load multi-view RGBD from a TRELLIS-style ``transforms.json`` directory."""
    with open(os.path.join(data_dir, "transforms.json")) as f:
        metadata = json.load(f)

    n_views_total = len(metadata["frames"])
    if num_views <= 0 or num_views > n_views_total:
        num_views = n_views_total
    view_indices = np.linspace(0, n_views_total - 1, num_views, dtype=int)

    images, depths, alphas, camera_params = [], [], [], []
    for view_idx in view_indices:
        frame = metadata["frames"][view_idx]
        rgb_filename = os.path.basename(frame["file_path"])
        depth_filename = rgb_filename.replace("_color.png", "_depth.png")
        rgb_image = Image.open(os.path.join(data_dir, rgb_filename)).convert("RGBA")
        images.append(rgb_image)
        depth_image = Image.open(os.path.join(data_dir, depth_filename))
        depths.append(np.array(depth_image).astype(np.float32) / 65535.0)
        alphas.append(np.array(rgb_image)[:, :, 3].astype(np.float32))
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


def _probs_to_rgb(probs: np.ndarray, colormap: bool = False) -> np.ndarray:
    """Map probabilities in [0, 1] to RGB.

    Default: grayscale (pixel intensity = probability). Higher prob =
    brighter pixel. Pass ``colormap=True`` to use viridis instead.
    """
    p = np.clip(probs, 0.0, 1.0)
    if colormap:
        try:
            import matplotlib

            rgba = matplotlib.colormaps["viridis"](p)
            return (rgba[..., :3] * 255).astype(np.uint8)
        except Exception:
            pass
    gray = (p * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def save_voxel_ply(path: str, coords: np.ndarray, probs: np.ndarray, resolution: int):
    """Write a colored .ply point cloud at voxel centers (cube space [-0.5, 0.5])."""
    centers = (coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    # Use viridis on the point cloud where colour helps separate magnitudes.
    colors = _probs_to_rgb(probs, colormap=True)
    n = centers.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(centers, colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def main(
    data_dir: str = "examples/data/sample2",
    query: str = None,
    num_views: int = 4,
    output_dir: str = "example_results",
    seed: int = 1,
    steps: int = 50,
    cfg_strength: float = 3.0,
    noise_scale: float = None,
    view_selection: str = "voxel",
    view_selection_transforms: str = None,
):
    """
    Run reconstruction + text-conditioned affordance grounding.

    Args:
        data_dir: Path to RGBD data directory (transforms.json format)
        query: Affordance text query. Defaults to the first Affogato
            test-split query for the bundled sample at ``data_dir``
            (see SAMPLE_QUERIES).
        num_views: Number of views to use (-1 for all)
        output_dir: Where to write outputs
        seed: Random seed
        steps: Affordance flow Euler steps
        cfg_strength: Classifier-free guidance strength
        noise_scale: Override the training-time logit-space noise scale (5.0)
        view_selection: rendering method for the active-view stage.
            ``"voxel"`` (default) = per-pixel raycast, decoder-free.
            ``"mesh"`` = paper version (SLAT mesh decode + nvdiffrast).
            ``None`` skips the stage.
        view_selection_transforms: path to a dataset ``transforms.json``
            listing the candidate poses. Defaults to
            ``<data_dir>/candidate_transforms.json`` when present.
    """
    sample_name = os.path.basename(os.path.normpath(data_dir))
    if query is None:
        if sample_name not in SAMPLE_QUERIES:
            raise ValueError(
                f"--query not given and no default for {sample_name!r}. "
                f"Known samples: {sorted(SAMPLE_QUERIES)}"
            )
        query = SAMPLE_QUERIES[sample_name][0]
        print(f"Using default query for {sample_name}: {query!r}")
        alt = SAMPLE_QUERIES[sample_name][1:]
        if alt:
            print("Other Affogato test-split queries for this sample:")
            for q in alt:
                print(f"  - {q}")

    print("=" * 60)
    print(f"Affostruction example  (query={query!r})")
    print("=" * 60)

    pipeline = AffostructionPipeline.from_pretrained().cuda()
    if view_selection is not None and view_selection_transforms is None:
        default_transforms = os.path.join(data_dir, "candidate_transforms.json")
        if os.path.isfile(default_transforms):
            view_selection_transforms = default_transforms
            print(f"Using candidate poses from: {view_selection_transforms}")
        else:
            raise FileNotFoundError(
                f"view_selection requires a transforms.json. None found at "
                f"{default_transforms!r}; pass --view_selection_transforms."
            )

    print(f"\nLoading RGBD data from: {data_dir}")
    input_dict = load_rgbd_data(data_dir, num_views=num_views)
    print(f"Loaded {len(input_dict['images'])} views")

    print("\nRunning reconstruction + affordance...")
    aff_params = {"steps": steps, "cfg_strength": cfg_strength, "seed": seed}
    if noise_scale is not None:
        aff_params["noise_scale"] = noise_scale
    formats = ["mesh"] if view_selection == "mesh" else None
    outputs = pipeline.run(
        input_dict,
        queries=[query],
        seed=seed,
        affordance_sampler_params=aff_params,
        view_selection_mode=view_selection,
        view_selection_transforms_path=view_selection_transforms,
        formats=formats,
    )

    coords = outputs["coords"].detach().cpu().numpy()
    probs = outputs["affordance"][0]["probs"].detach().cpu().numpy()
    voxel_resolution = int(
        pipeline.reconstruction_pipeline.models["slat_flow_model"].resolution
    )

    out_dir = Path(output_dir) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_query = query.replace(" ", "_")
    ply_path = out_dir / f"affordance_{safe_query}.ply"
    npz_path = out_dir / f"affordance_{safe_query}.npz"

    save_voxel_ply(str(ply_path), coords[:, 1:], probs, resolution=voxel_resolution)
    np.savez(
        str(npz_path),
        coords=coords,
        probs=probs,
        query=query,
        resolution=voxel_resolution,
    )

    print(f"\nSaved {coords.shape[0]} voxels.")
    print(f"  probability summary: min={probs.min():.3f}  "
          f"mean={probs.mean():.3f}  max={probs.max():.3f}")
    top_k = min(8, probs.shape[0])
    top_idx = np.argpartition(-probs, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]
    print(f"  top-{top_k} voxels (coord, prob):")
    for idx in top_idx:
        c = coords[idx, 1:].tolist()
        print(f"    coord={c}  prob={probs[idx]:.3f}")
    print(f"\nOutputs:\n  - {ply_path}\n  - {npz_path}")

    if view_selection is not None:
        vs = outputs["view_selection"][0]
        print(
            f"\nNext-best view (mode={vs['mode']}): "
            f"index={vs['index']}/{vs['scores'].shape[0]}  "
            f"yaw={np.degrees(vs['yaw']):.1f}°  pitch={np.degrees(vs['pitch']):.1f}°  "
            f"score={vs['score']:.1f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
