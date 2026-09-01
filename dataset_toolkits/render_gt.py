#!/usr/bin/env python3
"""
Render ground-truth views for reconstruction evaluation.

RGB comes from Blender CYCLES, the normal map from the TRELLIS renderer.

Camera parameters:
- Resolution: 512x512
- Radius: 2 (fixed)
- FOV: 40 degrees (fixed)
- Yaw: random ∈ [0, 2π)
- Pitch: random ∈ [-π/4, π/4] (reasonable viewing angles)
- Background: black (0, 0, 0) for RGB, gray (128, 128, 128) for normal

Output structure:
    recon_renders/{sha256}/
        rgb.png      - RGB rendering with lighting (Blender)
        normal.png   - Normal map (world-space, TRELLIS renderer)
        camera.txt   - Camera parameters "yaw pitch" in radians
                       (can be used with render_snapshot(offset=(yaw, pitch)))
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["SPCONV_ALGO"] = "native"

import torch
import trimesh
from PIL import Image
from affostruction.representations import MeshExtractResult
from affostruction.utils import render


def load_gt_mesh(mesh_path, device="cuda"):
    """
    Load GT mesh from PLY file and create MeshExtractResult for TRELLIS rendering.

    Args:
        mesh_path: Path to PLY file (colored mesh)
        device: Device to place tensors

    Returns:
        MeshExtractResult object ready for TRELLIS rendering
    """
    mesh = trimesh.load(mesh_path, process=False)

    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).long().to(device)


    bbox_min = vertices.min(dim=0)[0]
    bbox_max = vertices.max(dim=0)[0]

    center = (bbox_min + bbox_max) / 2
    vertices = vertices - center

    scale = (bbox_max - bbox_min).max()
    vertices = vertices / scale

    mesh_result = MeshExtractResult(
        vertices=vertices,
        faces=faces,
        vertex_attrs=None,
        res=64,
    )

    return mesh_result


def render_normal_with_trellis(
    mesh_path, yaw, pitch, output_path, r=2.0, fov=40.0, resolution=512, device="cuda"
):
    """
    Render normal map using TRELLIS render.

    This ensures GT and Pred normals are rendered with exactly the same renderer.

    Args:
        mesh_path: Path to PLY file
        yaw: Camera yaw in radians
        pitch: Camera pitch in radians
        output_path: Output path for normal.png
        r: Camera radius
        fov: Field of view in degrees
        resolution: Render resolution
        device: Device to use

    Returns:
        True if successful, False otherwise
    """
    try:
        mesh = load_gt_mesh(mesh_path, device=device)

        extrinsics, intrinsics = render.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [yaw], [pitch], [r], [fov]
        )

        results = render.render_frames(
            mesh, extrinsics, intrinsics, options={"resolution": resolution}, verbose=False
        )

        normal_img = results["normal"][0]
        Image.fromarray(normal_img).save(output_path)

        return True

    except Exception as e:
        print(f"[ERROR] TRELLIS normal rendering failed: {e}")
        return False


def generate_random_view(r=2, fov=40, seed=None):
    """
    Generate a random camera view with fixed radius and FOV.

    Args:
        r: Camera radius from origin (default: 2)
        fov: Field of view in degrees (default: 40)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Dict with camera parameters: {yaw, pitch, radius, fov}
    """
    if seed is not None:
        np.random.seed(seed)

    yaw = np.random.uniform(0, 2 * np.pi)

    pitch = np.random.uniform(-np.pi / 4, np.pi / 4)

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "radius": float(r),
        "fov": float(np.deg2rad(fov)),
    }


def render_sample(
    blend_file,
    sha256,
    output_dir,
    blender_path,
    render_script,
    colored_mesh_dir,
    resolution=512,
    r=2,
    fov=40,
    seed=None,
    debug=False,
    device="cuda",
):
    """
    Render a single sample with random camera view using hybrid rendering.

    Rendering pipeline:
    1. Blender: Render RGB with lighting + save camera.txt
    2. TRELLIS: Render normal using same camera parameters

    Args:
        blend_file: Path to .blend file
        sha256: SHA256 hash identifier
        output_dir: Base output directory
        blender_path: Path to Blender executable
        render_script: Path to Blender render script
        colored_mesh_dir: Directory containing colored mesh PLY files
        resolution: Image resolution (default 512)
        r: Camera radius (default 2)
        fov: Field of view in degrees (default 40)
        seed: Random seed (uses sha256 hash if None)
        debug: Enable verbose debug output
        device: Device for TRELLIS rendering (default 'cuda')

    Returns:
        Dict with sha256 and status
    """
    sample_output_dir = Path(output_dir) / sha256
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        print(f"\n{'='*80}")
        print(f"DEBUG: Rendering sample {sha256}")
        print(f"{'='*80}")
        print(f"Blend file: {blend_file}")
        print(f"Output dir: {sample_output_dir}")

    rgb_path = sample_output_dir / "rgb.png"
    normal_path = sample_output_dir / "normal.png"
    camera_path = sample_output_dir / "camera.txt"

    if rgb_path.exists() and normal_path.exists() and camera_path.exists():
        if debug:
            print("All outputs already exist, skipping...")
        return {"sha256": sha256, "status": "skipped", "message": "Already rendered"}

    if seed is None:
        seed = int(sha256[:8], 16) % (2**32)

    view = generate_random_view(r=r, fov=fov, seed=seed)
    view_json = json.dumps(view)

    if debug:
        print(f"\nCamera view (seed={seed}):")
        print(
            f"  yaw={view['yaw']:.4f} rad ({np.rad2deg(view['yaw']):.1f}°), "
            f"pitch={view['pitch']:.4f} rad ({np.rad2deg(view['pitch']):.1f}°)"
        )
        print(f"  radius={view['radius']}, fov={np.rad2deg(view['fov']):.1f}°")

    cmd = [
        blender_path,
        "--background",
        blend_file,
        "--python",
        render_script,
        "--",
        "--view",
        view_json,
        "--object",
        blend_file,
        "--output_folder",
        str(sample_output_dir),
        "--resolution",
        str(resolution),
        "--engine",
        "CYCLES",
    ]

    if debug:
        print(f"\nBlender command:")
        print(" ".join(cmd))
        print(f"\n{'='*80}")
        print("Running Blender (output will be shown below)...")
        print(f"{'='*80}\n")

    try:
        if debug:
            print("[INFO] Step 1/2: Rendering RGB with Blender...")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not debug else None,
            stderr=subprocess.PIPE if not debug else None,
            timeout=300,
            check=True,
        )

        if not (rgb_path.exists() and camera_path.exists()):
            missing = []
            if not rgb_path.exists():
                missing.append("rgb.png")
            if not camera_path.exists():
                missing.append("camera.txt")
            return {
                "sha256": sha256,
                "status": "error",
                "message": f"Blender failed to create: {', '.join(missing)}",
            }

        if debug:
            print(f"[INFO] Blender rendering complete: {rgb_path}")

        if debug:
            print("[INFO] Step 2/2: Rendering normal with TRELLIS...")

        with open(camera_path) as f:
            line = f.readline().strip()
            yaw, pitch = map(float, line.split())

        mesh_path = Path(colored_mesh_dir) / f"{sha256}.ply"
        if not mesh_path.exists():
            return {
                "sha256": sha256,
                "status": "error",
                "message": f"Colored mesh not found: {mesh_path}",
            }

        success = render_normal_with_trellis(
            mesh_path=mesh_path,
            yaw=yaw,
            pitch=pitch,
            output_path=normal_path,
            r=r,
            fov=fov,
            resolution=resolution,
            device=device,
        )

        if not success:
            return {
                "sha256": sha256,
                "status": "error",
                "message": "TRELLIS normal rendering failed",
            }

        if debug:
            print(f"[INFO] TRELLIS rendering complete: {normal_path}")

        if not (rgb_path.exists() and normal_path.exists() and camera_path.exists()):
            missing = []
            if not rgb_path.exists():
                missing.append("rgb.png")
            if not normal_path.exists():
                missing.append("normal.png")
            if not camera_path.exists():
                missing.append("camera.txt")
            return {
                "sha256": sha256,
                "status": "error",
                "message": f"Missing final outputs: {', '.join(missing)}",
            }

        if debug:
            print(f"[INFO] Hybrid rendering complete!")

        return {"sha256": sha256, "status": "success", "message": "Rendered successfully"}

    except subprocess.TimeoutExpired:
        return {"sha256": sha256, "status": "error", "message": "Blender timeout"}
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode()[:200] if e.stderr else "Unknown error"
        return {
            "sha256": sha256,
            "status": "error",
            "message": f"Blender error: {error_msg}",
        }
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {"sha256": sha256, "status": "error", "message": error_msg[:500]}


def load_test_samples(data_root, manifest):
    """
    Load Toys4k test split samples that have valid transforms.json.

    Args:
        data_root: Root directory containing toys4k data

    Returns:
        DataFrame with sha256 and local_path columns
    """
    metadata_path = Path(data_root) / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    with open(manifest) as f:
        selected = {line.split(",")[0].strip() for line in f if line.strip()}
    test_metadata = metadata[metadata["sha256"].isin(selected)].copy()

    print(f"Found {len(test_metadata)} of {len(selected)} manifest objects")

    valid_samples = []
    for _, row in test_metadata.iterrows():
        sha256 = row["sha256"]
        renders_cond_dir = Path(data_root) / "renders_cond" / sha256
        if renders_cond_dir.exists() and (renders_cond_dir / "transforms.json").exists():
            valid_samples.append({"sha256": sha256, "local_path": row["local_path"]})

    print(f"Found {len(valid_samples)} valid test samples with transforms.json")

    return pd.DataFrame(valid_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Render Toys4k ground truth with random views for reconstruction evaluation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="trellis_data/toys4k",
        help="Root directory for Toys4k dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmark/toys4k/gt/recon_renders",
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--blender-path",
        type=str,
        default="/tmp/blender-3.0.1-linux-x64/blender",
        help="Path to Blender executable",
    )
    parser.add_argument(
        "--render-script",
        type=str,
        default=None,
        help="Path to Blender render script (default: auto-detect)",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Image resolution (default: 512)"
    )
    parser.add_argument("--radius", type=float, default=2, help="Camera radius (default: 2)")
    parser.add_argument(
        "--fov", type=int, default=40, help="Field of view in degrees (default: 40)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples to render (for testing)"
    )
    parser.add_argument(
        "--debug-sha256",
        type=str,
        default=None,
        help="Debug a specific sample by SHA256 hash (renders only this sample with verbose output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for camera generation (default: use sha256 hash)",
    )
    parser.add_argument(
        "--colored-mesh-dir",
        type=str,
        default=None,
        help="Directory containing colored mesh PLY files (default: {data-root}/colored_meshes)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for TRELLIS rendering (default: cuda)",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Object list (default: manifests/{dataset}_test.txt)",
    )
    args = parser.parse_args()
    if args.manifest is None:
        manifests = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manifests"
        )
        args.manifest = os.path.join(manifests, f"{'toys4k'}_test.txt")

    if args.render_script is None:
        script_dir = Path(__file__).parent
        render_script = script_dir / "blender_script" / "render_gt.py"
        if not render_script.exists():
            raise FileNotFoundError(f"Render script not found: {render_script}")
        args.render_script = str(render_script)

    if args.colored_mesh_dir is None:
        args.colored_mesh_dir = str(Path(args.data_root) / "colored_meshes")

    print("Loading test samples...")
    test_samples = load_test_samples(args.data_root, args.manifest)

    if args.debug_sha256:
        test_samples = test_samples[test_samples["sha256"] == args.debug_sha256]
        if len(test_samples) == 0:
            print(f"ERROR: Sample with SHA256 '{args.debug_sha256}' not found in test split")
            return
        print(f"DEBUG MODE: Processing only sample {args.debug_sha256}")
        args.max_workers = 1
    elif args.limit:
        test_samples = test_samples.head(args.limit)
        print(f"Limited to {len(test_samples)} samples for testing")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering {len(test_samples)} samples with {args.max_workers} workers...")
    print(f"Rendering method: Hybrid (Blender RGB + TRELLIS Normal)")
    print(
        f"Camera parameters: r={args.radius}, fov={args.fov}°, resolution={args.resolution}x{args.resolution}"
    )
    print(f"Camera views: random yaw ∈ [0°, 360°), pitch ∈ [-45°, 45°]")
    print(f"Colored meshes: {args.colored_mesh_dir}")
    print(f"Output: {output_dir}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}

        for _, row in test_samples.iterrows():
            sha256 = row["sha256"]
            blend_file = Path(args.data_root) / row["local_path"]

            if not blend_file.exists():
                results.append(
                    {
                        "sha256": sha256,
                        "status": "error",
                        "message": f"Blend file not found: {blend_file}",
                    }
                )
                continue

            future = executor.submit(
                render_sample,
                str(blend_file),
                sha256,
                args.output_dir,
                args.blender_path,
                args.render_script,
                args.colored_mesh_dir,
                args.resolution,
                args.radius,
                args.fov,
                args.seed,
                debug=bool(args.debug_sha256),
                device=args.device,
            )
            futures[future] = sha256

        with tqdm(total=len(futures), desc="Rendering") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = result["status"]
                if status == "success":
                    pbar.set_postfix({"status": "success"})
                elif status == "skipped":
                    pbar.set_postfix({"status": "skipped"})
                else:
                    pbar.set_postfix({"error": result["message"][:30]})

                pbar.update(1)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("RENDERING SUMMARY")
    print("=" * 80)
    print(results_df["status"].value_counts().to_string())
    print("=" * 80)

    log_path = output_dir / "rendering_log.csv"
    results_df.to_csv(log_path, index=False)
    print(f"\nDetailed log saved to: {log_path}")

    errors = results_df[results_df["status"] == "error"]
    if len(errors) > 0:
        print(f"\n{len(errors)} errors occurred. First 10:")
        for _, row in errors.head(10).iterrows():
            print(f"  {row['sha256']}: {row['message']}")


if __name__ == "__main__":
    main()
