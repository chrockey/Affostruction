#!/usr/bin/env python3
"""
Sample point clouds from GT meshes for reconstruction evaluation.

This script:
1. Samples 100 camera views uniformly on a sphere (r=2, fov=40)
2. Renders depth + RGBA from each view using Blender
3. Unprojects depth to 3D points (alpha filtering)
4. Randomly samples 100k points from all views
5. Saves as PLY file

Output structure:
    points/{sha256}.ply - Sampled point cloud (100k points)
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from utils import sphere_hammersley_sequence

BLENDER_PATH = "/tmp/blender-3.0.1-linux-x64/blender"


def generate_camera_views(num_views=100, r=2.0, fov=40.0, seed=None):
    """
    Generate uniformly distributed camera views on a sphere.

    Args:
        num_views: Number of views to generate (default: 100)
        r: Camera radius (default: 2.0)
        fov: Field of view in degrees (default: 40)
        seed: Random seed for offset (default: None)

    Returns:
        List of camera view dicts with {yaw, pitch, radius, fov}
    """
    if seed is not None:
        np.random.seed(seed)

    offset = (np.random.rand(), np.random.rand())

    views = []
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        views.append(
            {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "radius": float(r),
                "fov": float(np.deg2rad(fov)),
            }
        )

    return views


def render_depth_rgba(
    blend_file,
    output_dir,
    views,
    resolution=512,
    blender_path=BLENDER_PATH,
    debug=False,
    timeout=1200,
    engine="BLENDER_EEVEE",
):
    """
    Render depth + RGBA images using Blender for multiple views.

    Args:
        blend_file: Path to .blend file
        output_dir: Directory to save rendered images
        views: List of camera view dicts
        resolution: Image resolution (default: 512)
        blender_path: Path to Blender executable
        debug: Enable verbose output
        timeout: Rendering timeout in seconds (default: 1200 = 20 min)
        engine: Blender engine to use (default: BLENDER_EEVEE)

    Returns:
        Dict with success status, duration, and error message if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blender_script = Path(__file__).parent / "blender_script" / "render.py"

    args = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24 +extension GLX"]

    if blend_file.endswith(".blend"):
        args.extend([blender_path, blend_file, "-b", "-P"])
    else:
        args.extend([blender_path, "-b", "-P"])

    args.extend(
        [
            str(blender_script),
            "--",
            "--views",
            json.dumps(views),
            "--object",
            os.path.expanduser(blend_file),
            "--resolution",
            str(resolution),
            "--output_folder",
            str(output_dir),
            "--engine",
            engine,
            "--save_depth",
        ]
    )

    if debug:
        print(f"Blender command: {' '.join(args)}")

    import time

    start_time = time.time()

    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE if not debug else None,
            stderr=subprocess.PIPE if not debug else None,
            timeout=timeout,
            check=True,
        )

        duration = time.time() - start_time

        transforms_path = output_dir / "transforms.json"
        if not transforms_path.exists():
            return {
                "success": False,
                "duration": duration,
                "error": "transforms.json not found",
                "timeout": False,
            }

        return {"success": True, "duration": duration, "error": None, "timeout": False}

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "error": f"Timeout after {timeout}s",
            "timeout": True,
        }
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        error_msg = e.stderr.decode()[:500] if e.stderr else "Unknown error"
        return {
            "success": False,
            "duration": duration,
            "error": f"Blender failed: {error_msg}",
            "timeout": False,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "duration": duration,
            "error": f"Exception: {str(e)[:500]}",
            "timeout": False,
        }


def unproject_depth(
    depth_array, alpha_mask, transform_matrix, depth_min, depth_max, camera_angle_x, height, width
):
    """
    Unproject depth image to 3D world coordinates.

    Args:
        depth_array: [H, W] normalized depth values [0, 1]
        alpha_mask: [H, W] alpha channel (0-1)
        transform_matrix: [4, 4] camera-to-world matrix (NeRF style)
        depth_min, depth_max: depth range
        camera_angle_x: camera field of view
        height, width: image dimensions

    Returns:
        xyz_world: [N, 3] world coordinates of valid points
    """
    absolute_depth = depth_array * (depth_max - depth_min) + depth_min

    valid_mask = (absolute_depth > 0) & (alpha_mask > 0)
    if not valid_mask.any():
        return np.zeros((0, 3), dtype=np.float32)

    focal_length = width / (2.0 * np.tan(camera_angle_x / 2.0))
    cx, cy = width / 2.0, height / 2.0

    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    x_cam = (x_coords - cx) * absolute_depth / focal_length
    y_cam = (y_coords - cy) * absolute_depth / focal_length
    z_cam = absolute_depth

    xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    xyz_cam_valid = xyz_cam[valid_mask]

    c2w = np.array(transform_matrix)
    c2w_corrected = c2w.copy()
    c2w_corrected[:3, 1:3] *= -1

    xyz_cam_homo = np.concatenate([xyz_cam_valid, np.ones((xyz_cam_valid.shape[0], 1))], axis=1)
    xyz_world_homo = xyz_cam_homo @ c2w_corrected.T
    xyz_world = xyz_world_homo[:, :3]

    bounds_mask = (xyz_world >= -0.5) & (xyz_world <= 0.5)
    bounds_mask = bounds_mask.all(axis=1)
    xyz_world = xyz_world[bounds_mask]

    return xyz_world.astype(np.float32)


def process_sample(
    blend_file,
    sha256,
    output_dir,
    num_views=100,
    num_points=100000,
    r=2.0,
    fov=40.0,
    resolution=512,
    seed=None,
    debug=False,
    engine="BLENDER_EEVEE",
    dataset="toys4k",
    data_root=None,
):
    """
    Process a single sample: render, unproject, sample points, save PLY.

    Args:
        blend_file: Path to .blend or .glb file
        sha256: Sample identifier
        output_dir: Base output directory for points
        num_views: Number of camera views (default: 100)
        num_points: Number of points to sample (default: 100k)
        r: Camera radius (default: 2.0)
        fov: Field of view in degrees (default: 40)
        resolution: Render resolution (default: 512)
        seed: Random seed (uses sha256 if None)
        debug: Enable verbose output
        engine: Blender render engine (default: BLENDER_EEVEE)
        dataset: Dataset type ('toys4k' or 'affogato')
        data_root: Root directory of dataset (required for 'affogato' to locate renders_cond)

    Returns:
        Dict with sha256 and status
    """
    output_path = Path(output_dir) / f"{sha256}.ply"

    if output_path.exists():
        return {"sha256": sha256, "status": "skipped", "message": "Already exists"}

    if debug:
        print(f"\n{'='*80}")
        print(f"Processing sample: {sha256}")
        print(f"{'='*80}")

    if seed is None:
        seed = int(sha256[:8], 16) % (2**32)

    try:
        if dataset == "affogato":
            if data_root is None:
                return {
                    "sha256": sha256,
                    "status": "error",
                    "message": "data_root is required for Affogato dataset",
                }

            mesh_file = Path(data_root) / "renders_cond" / sha256 / "mesh.ply"
            if not mesh_file.exists():
                return {
                    "sha256": sha256,
                    "status": "error",
                    "message": f"mesh.ply not found in renders_cond: {mesh_file}",
                }

            render_file = str(mesh_file)
            if debug:
                print(f"Using Affogato mesh from renders_cond: {mesh_file}")
        else:
            render_file = blend_file

        if debug:
            print(f"[1/4] Generating {num_views} camera views (seed={seed})...")
        views = generate_camera_views(num_views=num_views, r=r, fov=fov, seed=seed)

        if debug:
            print(f"[2/4] Rendering {num_views} views with Blender...")

        render_dir = Path("/tmp") / f"render_{sha256}"
        if render_dir.exists():
            import shutil

            shutil.rmtree(render_dir)

        timeout = 1800 if engine == "CYCLES" else 1200
        render_result = render_depth_rgba(
            blend_file=render_file,
            output_dir=render_dir,
            views=views,
            resolution=resolution,
            debug=debug,
            timeout=timeout,
            engine=engine,
        )

        if not render_result["success"] and render_result["timeout"] and engine == "BLENDER_EEVEE":
            if debug:
                print(
                    f"  EEVEE timed out ({render_result['duration']:.1f}s), retrying with CYCLES..."
                )

            if render_dir.exists():
                import shutil

                shutil.rmtree(render_dir)

            render_result = render_depth_rgba(
                blend_file=render_file,
                output_dir=render_dir,
                views=views,
                resolution=resolution,
                debug=debug,
                timeout=1800,
                engine="CYCLES",
            )

        if not render_result["success"]:
            return {
                "sha256": sha256,
                "status": "error",
                "message": render_result["error"],
                "duration": render_result["duration"],
                "timeout": render_result["timeout"],
            }

        if debug:
            print(f"[3/4] Unprojecting depth to 3D points...")

        transforms_path = render_dir / "transforms.json"
        with open(transforms_path) as f:
            transforms = json.load(f)

        all_points = []
        for frame_idx, frame in enumerate(transforms["frames"]):
            rgba_path = render_dir / frame["file_path"]
            rgba = Image.open(rgba_path)
            alpha = np.array(rgba.getchannel(3)).astype(np.float32) / 255.0

            depth_filename = Path(frame["file_path"]).stem + "_depth.png"
            depth_path = render_dir / depth_filename
            depth = Image.open(depth_path)
            depth_array = np.array(depth).astype(np.float32) / 65535.0

            xyz = unproject_depth(
                depth_array=depth_array,
                alpha_mask=alpha,
                transform_matrix=frame["transform_matrix"],
                depth_min=frame["depth"]["min"],
                depth_max=frame["depth"]["max"],
                camera_angle_x=frame["camera_angle_x"],
                height=depth_array.shape[0],
                width=depth_array.shape[1],
            )

            if xyz.shape[0] > 0:
                all_points.append(xyz)

            if debug and frame_idx % 20 == 0:
                print(f"  View {frame_idx}/{len(transforms['frames'])}: {xyz.shape[0]} points")

        if len(all_points) == 0:
            return {"sha256": sha256, "status": "error", "message": "No valid points"}

        all_points = np.concatenate(all_points, axis=0)

        if debug:
            print(f"  Total points from all views: {all_points.shape[0]}")

        if debug:
            print(f"[4/4] Sampling {num_points} points...")

        if all_points.shape[0] > num_points:
            np.random.seed(seed)
            indices = np.random.choice(all_points.shape[0], num_points, replace=False)
            sampled_points = all_points[indices]
        else:
            sampled_points = all_points
            if debug:
                print(f"  Warning: Only {all_points.shape[0]} points available (< {num_points})")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        point_cloud = trimesh.PointCloud(sampled_points)
        point_cloud.export(str(output_path))

        if debug:
            print(f"  Saved {sampled_points.shape[0]} points to: {output_path}")

        return {
            "sha256": sha256,
            "status": "success",
            "num_points": sampled_points.shape[0],
            "render_duration": render_result["duration"],
        }

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {"sha256": sha256, "status": "error", "message": error_msg[:500]}


def load_test_samples(data_root, manifest, dataset="toys4k"):
    """
    Load test split samples for the specified dataset.

    Args:
        data_root: Root directory containing dataset
        dataset: Dataset name ('toys4k' or 'affogato')

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

    print(f"Found {len(test_metadata)} of {len(selected)} manifest objects in {dataset}")

    return test_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Sample point clouds from GT meshes for reconstruction evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="toys4k",
        choices=["toys4k", "affogato"],
        help="Dataset to process (default: toys4k)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for dataset (auto-set based on --dataset if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for point cloud PLY files (auto-set based on --dataset if not provided)",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=100,
        help="Number of camera views to render (default: 100)",
    )
    parser.add_argument(
        "--num-points", type=int, default=100000, help="Number of points to sample (default: 100k)"
    )
    parser.add_argument("--radius", type=float, default=2.0, help="Camera radius (default: 2.0)")
    parser.add_argument(
        "--fov", type=int, default=40, help="Field of view in degrees (default: 40)"
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Render resolution (default: 512)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--debug-sha256",
        type=str,
        default=None,
        help="Debug a specific sample by SHA256 (verbose output)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: use sha256 hash)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Blender render engine (default: BLENDER_EEVEE)",
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
        args.manifest = os.path.join(manifests, f"{args.dataset}_test.txt")

    if args.data_root is None:
        if args.dataset == "toys4k":
            args.data_root = "trellis_data/toys4k"
        elif args.dataset == "affogato":
            args.data_root = "trellis_data/affogato"

    if args.output_dir is None:
        if args.dataset == "toys4k":
            args.output_dir = "results/benchmark/toys4k/gt/points"
        elif args.dataset == "affogato":
            args.output_dir = "results/benchmark/affogato/gt/points"

    print("Loading test samples...")
    test_samples = load_test_samples(args.data_root, args.manifest, dataset=args.dataset)

    if args.debug_sha256:
        test_samples = test_samples[test_samples["sha256"] == args.debug_sha256]
        if len(test_samples) == 0:
            print(f"ERROR: Sample '{args.debug_sha256}' not found in test split")
            return
        print(f"DEBUG MODE: Processing only sample {args.debug_sha256}")
        use_parallel = False
    else:
        if args.limit:
            test_samples = test_samples.head(args.limit)
            print(f"Limited to {len(test_samples)} samples")
        use_parallel = True

    print(f"\nProcessing {len(test_samples)} samples...")
    print(f"Camera: {args.num_views} views, r={args.radius}, fov={args.fov} degrees")
    print(f"Points: {args.num_points} sampled per mesh")
    print(f"Workers: {args.max_workers if use_parallel else 1} (parallel={use_parallel})")
    print(f"Output: {args.output_dir}\n")

    results = []
    log_path = Path(args.output_dir) / "sampling_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if use_parallel:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}

            for _, row in test_samples.iterrows():
                sha256 = row["sha256"]
                blend_file = Path(args.data_root) / row["local_path"]

                if args.dataset != "affogato" and not blend_file.exists():
                    results.append(
                        {
                            "sha256": sha256,
                            "status": "error",
                            "message": f"Blend file not found: {blend_file}",
                        }
                    )
                    continue

                future = executor.submit(
                    process_sample,
                    blend_file=str(blend_file),
                    sha256=sha256,
                    output_dir=args.output_dir,
                    num_views=args.num_views,
                    num_points=args.num_points,
                    r=args.radius,
                    fov=args.fov,
                    resolution=args.resolution,
                    seed=args.seed,
                    debug=False,
                                    engine=args.engine,
                    dataset=args.dataset,
                    data_root=args.data_root,
                )
                futures[future] = sha256

            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in as_completed(futures):
                    sha256 = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        if len(results) % 10 == 0:
                            pd.DataFrame(results).to_csv(log_path, index=False)

                        status = result["status"]
                        if status == "success":
                            pbar.set_postfix({"last": sha256[:8], "status": "OK"})
                        elif status == "skipped":
                            pbar.set_postfix({"last": sha256[:8], "status": "skip"})
                        else:
                            is_timeout = result.get("timeout", False)
                            if is_timeout:
                                tqdm.write(
                                    f"[WARN] Timeout on {sha256[:8]}: {result.get('message', 'Unknown')}"
                                )
                            pbar.set_postfix(
                                {"last": sha256[:8], "status": "TMO" if is_timeout else "ERR"}
                            )
                    except Exception as e:
                        results.append(
                            {
                                "sha256": sha256,
                                "status": "error",
                                "message": f"Exception: {str(e)[:100]}",
                            }
                        )
                        pbar.set_postfix({"last": sha256[:8], "status": "ERR"})

                    pbar.update(1)
    else:
        for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Processing"):
            sha256 = row["sha256"]
            blend_file = Path(args.data_root) / row["local_path"]

            if args.dataset != "affogato" and not blend_file.exists():
                results.append(
                    {
                        "sha256": sha256,
                        "status": "error",
                        "message": f"Blend file not found: {blend_file}",
                    }
                )
                continue

            result = process_sample(
                blend_file=str(blend_file),
                sha256=sha256,
                output_dir=args.output_dir,
                num_views=args.num_views,
                num_points=args.num_points,
                r=args.radius,
                fov=args.fov,
                resolution=args.resolution,
                seed=args.seed,
                debug=bool(args.debug_sha256),
                engine=args.engine,
                dataset=args.dataset,
                data_root=args.data_root,
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(results_df["status"].value_counts().to_string())
    print("=" * 80)

    success_results = results_df[results_df["status"] == "success"]
    if len(success_results) > 0 and "render_duration" in success_results.columns:
        print("\nRENDERING TIMING STATS (successful samples):")
        print(f"  Mean:   {success_results['render_duration'].mean():.1f}s")
        print(f"  Median: {success_results['render_duration'].median():.1f}s")
        print(f"  Min:    {success_results['render_duration'].min():.1f}s")
        print(f"  Max:    {success_results['render_duration'].max():.1f}s")
        print("=" * 80)

    results_df.to_csv(log_path, index=False)
    print(f"\nLog saved to: {log_path}")

    errors = results_df[results_df["status"] == "error"]
    if len(errors) > 0:
        if "timeout" in errors.columns:
            timeouts = errors[errors["timeout"] == True]
            other_errors = errors[errors["timeout"] != True]
        else:
            timeouts = pd.DataFrame()
            other_errors = errors

        if len(timeouts) > 0:
            print(f"\n{len(timeouts)} timeout errors occurred. First 5:")
            for _, row in timeouts.head(5).iterrows():
                duration = row.get("duration", 0)
                print(f"  {row['sha256'][:8]}: {row.get('message', 'Unknown')} ({duration:.0f}s)")

        if len(other_errors) > 0:
            print(f"\n{len(other_errors)} other errors occurred. First 5:")
            for _, row in other_errors.head(5).iterrows():
                print(f"  {row['sha256'][:8]}: {row.get('message', 'Unknown')[:80]}")


if __name__ == "__main__":
    main()
