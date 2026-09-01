"""
Unposed Real-World Reconstruction Example (experimental)

Single-view recovery without camera extrinsics — useful for in-the-wild
RGB-D snapshots, but the scale + pose heuristics are best-effort and
will misbehave when the view is far from the canonical front view or
when the object's in-plane AABB is a poor proxy for its full extent.

Runs AffostructionPipeline on a cropped RGB-D observation that has real
metric depth (.npy) and a pinhole intrinsics matrix but no camera
extrinsics. A canonical Blender Z-up "front view" pose is assumed (camera
at world origin, facing +Y, with +Z up); the pipeline normalizes the
unprojected point cloud into the TRELLIS canonical cube using the metric
depth to set object scale. The pose is chosen so that after the pipeline's
internal Blender->OpenCV Y/Z flip, image-y maps to TRELLIS down (-Z) and
OpenCV depth (+z_cam) maps to TRELLIS +Y. In Blender's default front view
(viewer at -Y looking toward +Y), the close-to-camera center of the object
appears closer to the viewer (convex/outside), matching how the training
data is oriented.

Expected data layout (see examples/data/sample3):
    crop_color.png           # RGB crop, HxW
    crop_mask.png            # object mask in the crop, HxW (nonzero = fg)
    crop_depth.npy           # float32 metric depth in meters, HxW
    original_mask.png        # object mask in the original (uncropped) frame
    original_intrinsics.json # {fx, fy, cx, cy, width, height} for the
                             # original frame; crop intrinsics are derived
                             # on the fly by matching the two masks.

Usage:
    python examples/reconstruction_unposed.py --data_dir examples/data/sample3
"""

import os

os.environ.setdefault("SPCONV_ALGO", "native")

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json

import fire
import numpy as np
from PIL import Image

from affostruction import AffostructionPipeline
from examples.reconstruction import save_outputs


def _derive_crop_intrinsics(
    original_intrinsics: dict,
    original_mask: np.ndarray,
    crop_mask: np.ndarray,
) -> dict:
    """
    Recover the square crop region used to produce `crop_mask` from
    `original_mask`, then map the original pinhole intrinsics into the crop
    frame. Assumes the crop was an axis-aligned square centered on the
    object (the object's bbox center in the original) and isotropically
    resized to the crop resolution.

    The square region size S is found by brute-forcing integer pixel sizes
    around the bbox-ratio estimate and picking the S that maximizes
    mask IoU between `resize(original_mask[region], crop_size)` and
    `crop_mask`. With pixel-perfect masks the search recovers S exactly.
    """
    orig_H, orig_W = original_mask.shape
    crop_H, crop_W = crop_mask.shape

    ys_o, xs_o = np.where(original_mask > 0)
    ys_c, xs_c = np.where(crop_mask > 0)
    if xs_o.size == 0 or xs_c.size == 0:
        raise ValueError("Empty mask; cannot derive crop region.")

    orig_obj_cx = (xs_o.min() + xs_o.max() + 1) / 2.0
    orig_obj_cy = (ys_o.min() + ys_o.max() + 1) / 2.0
    scale_guess = (xs_c.max() - xs_c.min() + 1) / (xs_o.max() - xs_o.min() + 1)
    s_guess = int(round(crop_W / scale_guess))

    best_iou, best_rect = -1.0, None
    for S in range(max(1, s_guess - 30), s_guess + 31):
        left = int(round(orig_obj_cx - S / 2.0))
        top = int(round(orig_obj_cy - S / 2.0))
        if left < 0 or top < 0 or left + S > orig_W or top + S > orig_H:
            continue
        region = original_mask[top : top + S, left : left + S]
        resized = np.array(
            Image.fromarray(region).resize((crop_W, crop_H), Image.NEAREST)
        )
        inter = ((resized > 0) & (crop_mask > 0)).sum()
        union = ((resized > 0) | (crop_mask > 0)).sum()
        iou = float(inter) / float(union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_rect = iou, (left, top, S)

    if best_rect is None:
        raise ValueError("Could not find a valid crop region inside the original image.")

    left, top, S = best_rect
    scale = crop_W / S
    return {
        "fx": float(original_intrinsics["fx"]) * scale,
        "fy": float(original_intrinsics["fy"]) * scale,
        "cx": (float(original_intrinsics["cx"]) - left) * scale,
        "cy": (float(original_intrinsics["cy"]) - top) * scale,
    }


def load_unposed_metric_data(data_dir: str) -> dict:
    """
    Build an input_dict for the ReconstructionPipeline from an unposed
    real-world crop with metric depth and pinhole intrinsics.

    Returns:
        dict with the same schema as examples.reconstruction.load_rgbd_data,
        but with a single view and camera_params[0] flagged metric_depth=True.
    """
    data_dir = str(data_dir)

    rgb_path = os.path.join(data_dir, "crop_color.png")
    crop_mask_path = os.path.join(data_dir, "crop_mask.png")
    depth_path = os.path.join(data_dir, "crop_depth.npy")
    original_mask_path = os.path.join(data_dir, "original_mask.png")
    original_intr_path = os.path.join(data_dir, "original_intrinsics.json")

    rgb_image = Image.open(rgb_path).convert("RGB")
    mask_image = Image.open(crop_mask_path).convert("L").resize(
        rgb_image.size, Image.NEAREST
    )
    rgb_arr = np.array(rgb_image)
    mask_arr = np.array(mask_image)
    rgb_arr[mask_arr == 0] = 0
    rgba = Image.fromarray(
        np.concatenate([rgb_arr, mask_arr[..., None]], axis=-1),
        mode="RGBA",
    )

    alpha_array = mask_arr.astype(np.float32)

    depth_array = np.load(depth_path).astype(np.float32)
    if depth_array.shape[:2] != (rgba.size[1], rgba.size[0]):
        raise ValueError(
            f"Depth shape {depth_array.shape} does not match image size "
            f"{(rgba.size[1], rgba.size[0])}"
        )

    with open(original_intr_path) as f:
        original_intrinsics = json.load(f)
    for key in ("fx", "fy", "cx", "cy"):
        if key not in original_intrinsics:
            raise KeyError(f"{original_intr_path} missing required key '{key}'")
    original_mask = np.array(Image.open(original_mask_path).convert("L"))
    crop_mask = mask_arr
    intr = _derive_crop_intrinsics(original_intrinsics, original_mask, crop_mask)

    cam_params = {
        "intrinsics": {
            "fx": intr["fx"],
            "fy": intr["fy"],
            "cx": intr["cx"],
            "cy": intr["cy"],
        },
        "transform_matrix": np.array(
            [
                [1.0, 0.0,  0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0,  0.0, 0.0],
                [0.0, 0.0,  0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "metric_depth": True,
    }

    return {
        "images": [rgba],
        "depths": [depth_array],
        "alphas": [alpha_array],
        "camera_params": [cam_params],
    }


def main(
    data_dir: str = "examples/data/sample3",
    output_dir: str = "example_results",
    seed: int = 1,
    ss_steps: int = None,
    ss_cfg_strength: float = None,
    slat_steps: int = None,
    slat_cfg_strength: float = None,
    simplify: float = 0.95,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    render_video: bool = True,
):
    """
    Run 3D reconstruction on an unposed real-world crop with metric depth.
    """
    print("=" * 60)
    print("Affostruction - Unposed Reconstruction (metric depth)")
    print("=" * 60)

    pipeline = AffostructionPipeline.from_pretrained()
    pipeline.cuda()

    print(f"\nLoading unposed data from: {data_dir}")
    input_dict = load_unposed_metric_data(data_dir)
    print(
        f"  image size: {input_dict['images'][0].size}, "
        f"depth range: [{input_dict['depths'][0][input_dict['depths'][0] > 0].min():.3f}, "
        f"{input_dict['depths'][0].max():.3f}] m"
    )

    sample_name = os.path.basename(os.path.normpath(data_dir))
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)

    ss_params = {}
    if ss_steps is not None:
        ss_params["steps"] = ss_steps
    if ss_cfg_strength is not None:
        ss_params["cfg_strength"] = ss_cfg_strength

    slat_params = {}
    if slat_steps is not None:
        slat_params["steps"] = slat_steps
    if slat_cfg_strength is not None:
        slat_params["cfg_strength"] = slat_cfg_strength

    print("\nRunning reconstruction...")
    outputs = pipeline.run(
        input_dict,
        seed=seed,
        sparse_structure_sampler_params=ss_params if ss_params else None,
        slat_sampler_params=slat_params if slat_params else None,
        formats=["mesh", "gaussian"],
    )

    print(f"\nSaving outputs to: {sample_output_dir}")
    save_outputs(
        outputs,
        sample_output_dir,
        simplify=simplify,
        fill_holes_max_size=fill_holes_max_size,
        texture_size=texture_size,
        render_video=render_video,
    )

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)
