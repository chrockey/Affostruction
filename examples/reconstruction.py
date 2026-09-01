"""
Reconstruction Example

This script demonstrates how to use AffostructionPipeline for
3D reconstruction from multi-view RGBD images.

Usage:
    python examples/reconstruction.py --data_dir examples/data/sample1 --num_views 3

    # With custom sampler params
    python examples/reconstruction.py --data_dir examples/data/sample1 \
        --ss_steps 25 --ss_cfg_strength 5.0

    # With all available views
    python examples/reconstruction.py --data_dir examples/data/sample1 --num_views -1
"""

import os

os.environ.setdefault("SPCONV_ALGO", "native")

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from pathlib import Path

import fire
import imageio
import numpy as np
from PIL import Image

from affostruction import AffostructionPipeline
from affostruction.utils import postprocessing, render


def load_rgbd_data(data_dir: str, num_views: int = 3) -> dict:
    """
    Load RGBD data from transforms.json format directory.

    Args:
        data_dir: Path to directory containing transforms.json, RGB images, and depth maps
        num_views: Number of views to load (-1 for all views)

    Returns:
        dict with keys: images, depths, alphas, camera_params
    """
    with open(os.path.join(data_dir, "transforms.json")) as f:
        metadata = json.load(f)

    n_views_total = len(metadata["frames"])

    if num_views <= 0 or num_views > n_views_total:
        num_views = n_views_total

    view_indices = np.linspace(0, n_views_total - 1, num_views, dtype=int)

    images = []
    depths = []
    alphas = []
    camera_params = []

    for view_idx in view_indices:
        frame = metadata["frames"][view_idx]

        rgb_filename = os.path.basename(frame["file_path"])
        rgb_path = os.path.join(data_dir, rgb_filename)
        depth_filename = rgb_filename.replace("_color.png", "_depth.png")
        depth_path = os.path.join(data_dir, depth_filename)

        rgb_image = Image.open(rgb_path).convert("RGBA")
        images.append(rgb_image)

        depth_image = Image.open(depth_path)
        depth_array = np.array(depth_image).astype(np.float32) / 65535.0
        depths.append(depth_array)

        alpha_array = np.array(rgb_image)[:, :, 3].astype(np.float32)
        alphas.append(alpha_array)

        cam_params = {
            "transform_matrix": frame["transform_matrix"],
            "depth_min": frame["depth"]["min"],
            "depth_max": frame["depth"]["max"],
            "camera_angle_x": frame["camera_angle_x"],
        }
        camera_params.append(cam_params)

    return {
        "images": images,
        "depths": depths,
        "alphas": alphas,
        "camera_params": camera_params,
    }


def save_outputs(
    outputs: dict,
    output_dir: str,
    simplify: float = 0.95,
    fill_holes_max_size: float = 0.04,
    texture_size: int = 1024,
    render_video: bool = True,
):
    """
    Save reconstruction outputs.

    Args:
        outputs: Pipeline outputs with 'mesh' and 'gaussian' keys
        output_dir: Output directory
        simplify: Mesh simplification ratio
        fill_holes_max_size: Maximum hole size to fill
        texture_size: Texture size for GLB export
        render_video: Whether to render video
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mesh = outputs["mesh"][0]
    gaussian = outputs["gaussian"][0]

    print("Exporting GLB...")
    glb = postprocessing.to_glb(
        gaussian,
        mesh,
        simplify=simplify,
        texture_size=texture_size,
        fill_holes_max_size=fill_holes_max_size,
    )
    glb.export(str(output_path / "mesh.glb"))

    print("Saving Gaussian PLY...")
    gaussian.save_ply(str(output_path / "gaussian.ply"))

    if render_video:
        print("Rendering video...")
        video_gs = render.render_video(gaussian)["color"]
        video_mesh = render.render_video(mesh)["normal"]
        video = [
            np.concatenate([frame_gs, frame_mesh], axis=1)
            for frame_gs, frame_mesh in zip(video_gs, video_mesh)
        ]
        imageio.mimsave(str(output_path / "rendered.mp4"), video, fps=30)

    print(f"\nOutputs saved to: {output_path}")
    print(f"  - mesh.glb")
    print(f"  - gaussian.ply")
    if render_video:
        print(f"  - rendered.mp4")


def main(
    data_dir: str = "examples/data/sample1",
    num_views: int = 3,
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
    Run 3D reconstruction from multi-view RGBD.

    Args:
        data_dir: Path to RGBD data directory
        num_views: Number of views to use (-1 for all)
        output_dir: Output directory
        seed: Random seed
        ss_steps: Sparse structure sampler steps
        ss_cfg_strength: Sparse structure CFG strength
        slat_steps: SLAT sampler steps
        slat_cfg_strength: SLAT CFG strength
        simplify: Mesh simplification ratio
        fill_holes_max_size: Maximum hole size to fill
        texture_size: Texture size for GLB
        render_video: Whether to render video
    """
    print("=" * 60)
    print("Affostruction - 3D Reconstruction Example")
    print("=" * 60)

    pipeline = AffostructionPipeline.from_pretrained()
    pipeline.cuda()

    print(f"\nLoading RGBD data from: {data_dir}")
    input_dict = load_rgbd_data(data_dir, num_views)
    print(f"Loaded {len(input_dict['images'])} views")

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
