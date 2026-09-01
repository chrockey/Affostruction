#!/usr/bin/env python3
"""
Sample point clouds from predicted meshes for reconstruction evaluation.

This script:
1. Loads predicted mesh.glb files
2. Samples 100 camera views uniformly on a sphere (r=2, fov=40)
3. Renders depth + mask from each view using TRELLIS MeshRenderer
4. Unprojects depth to 3D points (mask filtering)
5. Randomly samples 100k points from all views
6. Saves as PLY file

Output structure:
    {dataset}/{model}/{num_views}/{sha256}/points.ply - Sampled point cloud (100k points)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import trimesh
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from affostruction.representations.mesh import MeshExtractResult
from affostruction.renderers.mesh_renderer import MeshRenderer
from affostruction.utils.render import yaw_pitch_r_fov_to_extrinsics_intrinsics

sys.path.insert(0, str(Path(__file__).parent))
from utils import sphere_hammersley_sequence


def load_glb_mesh(glb_path: str, device="cuda") -> MeshExtractResult:
    """
    Load a GLB file and convert it to MeshExtractResult format.

    Args:
        glb_path: Path to GLB file
        device: Device to load mesh to (default: 'cuda')

    Returns:
        MeshExtractResult object
    """
    mesh = trimesh.load(glb_path, force="mesh")

    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).long().to(device)

    vertex_attrs = None
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors[:, :3]).astype(np.float32) / 255.0
        vertex_attrs = torch.from_numpy(colors).float().to(device)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = torch.cross(edge1, edge2, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    mesh_result = MeshExtractResult(
        vertices=vertices,
        faces=faces,
        vertex_attrs=vertex_attrs,
    )

    mesh_result.face_normal = face_normals

    return mesh_result


def generate_camera_views(num_views=100, r=2.0, fov=40.0, seed=None):
    """
    Generate uniformly distributed camera views on a sphere.

    Args:
        num_views: Number of views to generate (default: 100)
        r: Camera radius (default: 2.0)
        fov: Field of view in degrees (default: 40)
        seed: Random seed for offset (default: None)

    Returns:
        Tuple of (yaws, pitches, rs, fovs) lists
    """
    if seed is not None:
        np.random.seed(seed)

    offset = (np.random.rand(), np.random.rand())

    yaws = []
    pitches = []
    rs = []
    fovs = []

    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(float(yaw))
        pitches.append(float(pitch))
        rs.append(float(r))
        fovs.append(float(fov))

    return yaws, pitches, rs, fovs


def unproject_depth(
    depth_tensor,
    mask_tensor,
    extrinsics,
    intrinsics,
    near=1.0,
    far=100.0,
):
    """
    Unproject depth from MeshRenderer output to 3D world coordinates.

    Args:
        depth_tensor: [H, W] depth tensor (camera-space Z coordinates)
        mask_tensor: [H, W] mask tensor (0-1)
        extrinsics: [4, 4] world-to-camera extrinsics matrix
        intrinsics: [3, 3] camera intrinsics matrix
        near: Near plane (for filtering)
        far: Far plane (for filtering)

    Returns:
        xyz_world: [N, 3] world coordinates of valid points
    """
    device = depth_tensor.device
    H, W = depth_tensor.shape

    valid_mask = (depth_tensor > near) & (depth_tensor < far) & (mask_tensor > 0.5)

    if not valid_mask.any():
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )

    z_cam = depth_tensor[valid_mask]
    x_pixel = x_coords[valid_mask].float()
    y_pixel = y_coords[valid_mask].float()

    x_cam = (x_pixel / W - cx) * z_cam / fx
    y_cam = (y_pixel / H - cy) * z_cam / fy

    xyz_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    c2w = torch.inverse(extrinsics)

    xyz_cam_homo = torch.cat([xyz_cam, torch.ones((xyz_cam.shape[0], 1), device=device)], dim=1)
    xyz_world_homo = xyz_cam_homo @ c2w.T
    xyz_world = xyz_world_homo[:, :3]

    bounds_mask = (xyz_world >= -0.5) & (xyz_world <= 0.5)
    bounds_mask = bounds_mask.all(dim=1)
    xyz_world = xyz_world[bounds_mask]

    return xyz_world


def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0, debug=False):
    """
    Drop points whose mean distance to their K nearest neighbours exceeds
    ``mean + std_ratio * std`` over the cloud, which removes isolated floaters.

    Args:
        points: [N, 3] numpy array
        nb_neighbors: Number of neighbours per point (default: 20)
        std_ratio: Threshold multiplier (default: 2.0)
        debug: Enable verbose output

    Returns:
        filtered_points: [M, 3] with outliers removed
        num_removed: number of points removed
    """
    if nb_neighbors <= 0:
        return points, 0

    if len(points) < nb_neighbors + 1:
        if debug:
            print(
                f"  Outlier removal skipped: not enough points ({len(points)} < {nb_neighbors + 1})"
            )
        return points, 0

    distances, _ = cKDTree(points).query(points, k=nb_neighbors + 1, workers=-1)
    avg_distances = distances[:, 1:].mean(axis=1)

    threshold = avg_distances.mean() + std_ratio * avg_distances.std(ddof=1)
    inliers = avg_distances < threshold
    num_removed = int((~inliers).sum())

    if debug:
        print(f"  Outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio}):")
        print(f"    Mean neighbor distance: {avg_distances.mean():.6f}")
        print(f"    Std neighbor distance: {avg_distances.std(ddof=1):.6f}")
        print(f"    Threshold: {threshold:.6f}")
        print(f"    Removed: {num_removed} / {len(points)} ({100 * num_removed / len(points):.3f}%)")

    return points[inliers], num_removed


def process_sample(
    mesh_glb_path,
    sha256,
    output_path,
    num_views=100,
    num_points=100000,
    r=2.0,
    fov=40.0,
    resolution=512,
    seed=None,
    debug=False,
    nb_neighbors=20,
    std_ratio=2.0,
    force=False,
):
    """
    Process a single predicted mesh: render, unproject, sample points, save PLY.

    Args:
        mesh_glb_path: Path to mesh.glb file
        sha256: Sample identifier
        output_path: Output path for points.ply
        num_views: Number of camera views (default: 100)
        num_points: Number of points to sample (default: 100k)
        r: Camera radius (default: 2.0)
        fov: Field of view in degrees (default: 40)
        resolution: Render resolution (default: 512)
        seed: Random seed (uses sha256 if None)
        debug: Enable verbose output
        nb_neighbors: Number of neighbors for outlier removal (0=disable, default: 20)
        std_ratio: Std threshold multiplier for outlier removal (default: 2.0)
        force: Force regeneration of existing files (default: False)

    Returns:
        Dict with sha256 and status
    """
    if not force and Path(output_path).exists():
        return {"sha256": sha256, "status": "skipped", "message": "Already exists"}

    if debug:
        print(f"\n{'='*80}")
        print(f"Processing sample: {sha256}")
        print(f"{'='*80}")

    if seed is None:
        seed = int(sha256[:8], 16) % (2**32)

    try:
        if debug:
            print(f"[1/5] Loading mesh from {mesh_glb_path}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        mesh = load_glb_mesh(str(mesh_glb_path), device=device)

        if debug:
            print(f"  Vertices: {mesh.vertices.shape[0]}, Faces: {mesh.faces.shape[0]}")

        if debug:
            print(f"[2/5] Generating {num_views} camera views (seed={seed})...")

        yaws, pitches, rs, fovs = generate_camera_views(
            num_views=num_views, r=r, fov=fov, seed=seed
        )

        extrinsics_list, intrinsics_list = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitches, rs, fovs
        )

        if debug:
            print(f"[3/5] Setting up MeshRenderer...")

        renderer = MeshRenderer(device=device)
        renderer.rendering_options.resolution = resolution
        renderer.rendering_options.near = 1.0
        renderer.rendering_options.far = 100.0
        renderer.rendering_options.ssaa = 1

        if debug:
            print(f"[4/5] Rendering {num_views} views with batch rendering...")

        extrinsics_batch = torch.stack(extrinsics_list)
        intrinsics_batch = torch.stack(intrinsics_list)

        render_output = renderer.render(
            mesh, extrinsics_batch, intrinsics_batch, return_types=["depth", "mask"]
        )

        depth_batch = render_output["depth"]
        mask_batch = render_output["mask"]

        if debug:
            print(f"  Rendered {num_views} views in batch")
            print(f"[5/6] Unprojecting {num_views} views...")

        all_points = []

        iterator = range(num_views)
        if not debug:
            iterator = tqdm(iterator, total=num_views, desc=f"  {sha256[:8]}", leave=False)

        for view_idx in iterator:
            xyz = unproject_depth(
                depth_tensor=depth_batch[view_idx],
                mask_tensor=mask_batch[view_idx],
                extrinsics=extrinsics_list[view_idx],
                intrinsics=intrinsics_list[view_idx],
                near=renderer.rendering_options.near,
                far=renderer.rendering_options.far,
            )

            if xyz.shape[0] > 0:
                all_points.append(xyz.cpu().numpy())

            if debug and view_idx % 20 == 0:
                print(f"  View {view_idx}/{num_views}: {xyz.shape[0]} points")

        if len(all_points) == 0:
            return {"sha256": sha256, "status": "error", "message": "No valid points"}

        all_points = np.concatenate(all_points, axis=0)

        if debug:
            print(f"  Total points from all views: {all_points.shape[0]}")

        if nb_neighbors > 0:
            if debug:
                print(f"[6/7] Removing statistical outliers...")

            all_points, num_removed = remove_statistical_outliers(
                all_points, nb_neighbors=nb_neighbors, std_ratio=std_ratio, debug=debug
            )

            if debug:
                print(f"  Points after outlier removal: {all_points.shape[0]}")

        if debug:
            print(f"[7/7] Sampling {num_points} points...")

        if all_points.shape[0] > num_points:
            np.random.seed(seed)
            indices = np.random.choice(all_points.shape[0], num_points, replace=False)
            sampled_points = all_points[indices]
        else:
            sampled_points = all_points
            if debug:
                print(f"  Warning: Only {all_points.shape[0]} points available (< {num_points})")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        point_cloud = trimesh.PointCloud(sampled_points)
        point_cloud.export(str(output_path))

        if debug:
            print(f"  Saved {sampled_points.shape[0]} points to: {output_path}")

        return {
            "sha256": sha256,
            "status": "success",
            "num_points": sampled_points.shape[0],
        }

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return {"sha256": sha256, "status": "error", "message": error_msg[:500]}


def load_samples_from_benchmark_results(
    output_base: Path,
    dataset_name: str,
    model_name: str,
    num_views: int,
):
    """
    Load samples from benchmark inference results.

    Args:
        output_base: Base output directory
        dataset_name: Dataset name (toys4k or affogato)
        model_name: Model name
        num_views: Number of views used

    Returns:
        List of (sha256, mesh_glb_path, output_ply_path) tuples
    """
    benchmark_dir = output_base / dataset_name / model_name / str(num_views)

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    samples = []

    for sha_dir in benchmark_dir.iterdir():
        if not sha_dir.is_dir():
            continue

        sha256 = sha_dir.name
        mesh_glb = sha_dir / "mesh.glb"
        output_ply = sha_dir / "points.ply"

        if mesh_glb.exists():
            samples.append((sha256, str(mesh_glb), str(output_ply)))

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Sample point clouds from predicted meshes for reconstruction evaluation"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results/benchmark",
        help="Base directory for benchmark outputs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (toys4k or affogato)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model tag used by predict.py (the subdir under the benchmark root)",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=1,
        help="Number of input views used for prediction (default: 8)",
    )
    parser.add_argument(
        "--sampling-views",
        type=int,
        default=100,
        help="Number of camera views for point sampling (default: 100)",
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
        "--nb-neighbors",
        type=int,
        default=20,
        help="Number of neighbors for statistical outlier removal (0=disable, default: 20)",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="Std ratio for outlier removal threshold (default: 2.0, lower=aggressive, higher=conservative)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing points.ply files (default: skip existing)",
    )

    args = parser.parse_args()

    print("Loading samples from benchmark results...")
    output_base = Path(args.output_base)

    samples = load_samples_from_benchmark_results(
        output_base=output_base,
        dataset_name=args.dataset,
        model_name=args.model,
        num_views=args.num_views,
    )

    print(f"Found {len(samples)} samples with mesh.glb files")

    if args.debug_sha256:
        samples = [s for s in samples if s[0] == args.debug_sha256]
        if len(samples) == 0:
            print(f"ERROR: Sample '{args.debug_sha256}' not found")
            return
        print(f"DEBUG MODE: Processing only sample {args.debug_sha256}")
        use_parallel = False
    elif args.limit:
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} samples")
        use_parallel = True
    else:
        use_parallel = True

    print(f"\nProcessing {len(samples)} samples...")
    print(
        f"Camera: {args.sampling_views} views (batch rendering), r={args.radius}, fov={args.fov} degrees"
    )
    print(f"Points: {args.num_points} sampled per mesh")
    print(f"Mode: Sequential (batch rendering per sample)")
    print(f"Output: {output_base / args.dataset / args.model / str(args.num_views)}\n")

    results = []
    log_dir = output_base / args.dataset / args.model / str(args.num_views)
    log_path = log_dir / "points_sampling_log.csv"

    iterator = samples
    if use_parallel:
        iterator = tqdm(samples, desc="Processing")

    for sha256, mesh_glb, output_ply in iterator:
        try:
            result = process_sample(
                mesh_glb_path=mesh_glb,
                sha256=sha256,
                output_path=output_ply,
                num_views=args.sampling_views,
                num_points=args.num_points,
                r=args.radius,
                fov=args.fov,
                resolution=args.resolution,
                seed=args.seed,
                debug=not use_parallel,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                force=args.force,
            )
            results.append(result)

            if len(results) % 10 == 0:
                pd.DataFrame(results).to_csv(log_path, index=False)

            if use_parallel:
                status = result["status"]
                if status == "success":
                    iterator.set_postfix({"last": sha256[:8], "status": "OK"})
                elif status == "skipped":
                    iterator.set_postfix({"last": sha256[:8], "status": "skip"})
                else:
                    iterator.set_postfix({"last": sha256[:8], "status": "ERR"})

        except Exception as e:
            results.append(
                {
                    "sha256": sha256,
                    "status": "error",
                    "message": f"Exception: {str(e)[:100]}",
                }
            )
            if use_parallel:
                iterator.set_postfix({"last": sha256[:8], "status": "ERR"})

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(results_df["status"].value_counts().to_string())
    print("=" * 80)

    results_df.to_csv(log_path, index=False)
    print(f"\nLog saved to: {log_path}")

    errors = results_df[results_df["status"] == "error"]
    if len(errors) > 0:
        print(f"\n{len(errors)} errors occurred. First 5:")
        for _, row in errors.head(5).iterrows():
            print(f"  {row['sha256'][:8]}: {row.get('message', 'Unknown')[:80]}")


if __name__ == "__main__":
    main()
