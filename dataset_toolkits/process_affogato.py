#!/usr/bin/env python3
"""
Process AffoGato dataset samples to TRELLIS format in batch.

This script converts the entire AffoGato dataset to the standard format:
- renders_cond/{sha256}/: 40 RGB images, 40 depth maps, mesh.ply, transforms.json
- voxels/{sha256}.ply: Voxelized point cloud (from affordance annotation points)
- affordances/{sha256}/affordance.npz: Voxel heatmaps and affordance queries
  - 'heatmap': (N_voxels, 5) float32 - 5 affordance heatmap channels
  - 'query': array of 5 text queries

IMPORTANT: Affordance annotations are REQUIRED for all samples.

Usage:
    # Single node processing
    python process_affogato.py Affo \
        --affogato_base_dir /path/to/affogato_raw \
        --output_dir ~/datasets/trellis/affogato \
        --max_workers 16

    # Multi-node processing (node 0 of 4)
    python process_affogato.py Affo \
        --affogato_base_dir /path/to/affogato_raw \
        --output_dir ~/datasets/trellis/affogato \
        --rank 0 --world_size 4 --max_workers 16
"""

import os
import sys
import time
import json
import re
import importlib
import argparse
import tarfile
from io import BytesIO
from pathlib import Path
from easydict import EasyDict as edict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from affogato_io import read_mesh_trimesh, read_depth, read_cam_params


def process_sample(
    file_identifier, sha256, tar_path, glb_path, output_dir, affogato_base_dir=None, force=False
):
    """
    Process a single AffoGato sample.

    Args:
        file_identifier: Object UID (Affogato UID, 32-char hex)
        sha256: SHA256 hash (from ObjaverseXL if available, otherwise SHA256(UID))
        tar_path: Path to tar.gz file with renders
        glb_path: Path to glb mesh file
        output_dir: Base output directory
        force: If True, overwrite existing files

    Returns:
        dict with processing results or None on failure
    """
    try:
        output_dir = Path(output_dir)
        renders_cond_dir = output_dir / "renders_cond" / sha256

        affordances_dir = output_dir / "affordances" / sha256
        if not force and (
            (renders_cond_dir / "transforms.json").exists()
            and (affordances_dir / "affordance.npz").exists()
        ):
            return None

        renders_cond_dir.mkdir(parents=True, exist_ok=True)

        mesh = read_mesh_trimesh(glb_path)

        bounds = mesh.bounding_box.bounds
        bounds_min = bounds[0]
        bounds_max = bounds[1]

        scale = 1.0 / max(bounds_max - bounds_min)

        offset = -(bounds_max + bounds_min) / 2

        mesh_normalized = mesh.copy()
        mesh_normalized.vertices = scale * (mesh_normalized.vertices + offset)

        coord_transform = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        )
        mesh_normalized.vertices = (coord_transform @ mesh_normalized.vertices.T).T

        mesh_ply_path = renders_cond_dir / "mesh.ply"
        mesh_normalized.export(str(mesh_ply_path))

        with tarfile.open(tar_path, "r:*") as tar:
            members = tar.getmembers()

            views_dict = {}
            for m in members:
                fname = os.path.basename(m.name)

                if fname.endswith(".json"):
                    view_id = fname[:-5]
                    views_dict.setdefault(view_id, {})["json"] = m
                elif fname.endswith("_nd.exr"):
                    view_id = fname[:-7]
                    views_dict.setdefault(view_id, {})["depth"] = m
                elif re.match(r"\d{5}\.png$", fname):
                    view_id = fname[:-4]
                    views_dict.setdefault(view_id, {})["image"] = m

            sorted_view_ids = sorted(views_dict.keys())
            images = []
            depths = []
            c2ws = []
            Ks = []

            for view_id in sorted_view_ids:
                view_files = views_dict[view_id]

                with tar.extractfile(view_files["image"]) as f:
                    img = Image.open(BytesIO(f.read()))
                    images.append(img)

                with tar.extractfile(view_files["depth"]) as f:
                    depth = read_depth(f)
                    depths.append(depth)

                with tar.extractfile(view_files["json"]) as f:
                    img_size = images[0].size
                    c2w, K = read_cam_params(f, img_size)

                    coord_transform_3x3 = np.array(
                        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
                    )

                    c2w_position = c2w[:3, 3]
                    c2w_position_transformed = coord_transform_3x3 @ c2w_position

                    c2w_rotation = c2w[:3, :3]
                    c2w_rotation_transformed = coord_transform_3x3 @ c2w_rotation

                    c2w_transformed = np.eye(4, dtype=np.float32)
                    c2w_transformed[:3, :3] = c2w_rotation_transformed
                    c2w_transformed[:3, 3] = c2w_position_transformed

                    c2w_transformed[:3, 1] *= -1
                    c2w_transformed[:3, 2] *= -1

                    c2ws.append(c2w_transformed)
                    Ks.append(K)

        num_views = len(images)

        for i, img in enumerate(images):
            img_path = renders_cond_dir / f"{i:03d}.png"
            img.save(img_path)

        depth_ranges = []

        for i, depth in enumerate(depths):
            valid_mask = depth > 0

            if np.any(valid_mask):
                img_array = np.array(images[i])
                if img_array.shape[2] == 4:
                    alpha_channel = img_array[:, :, 3]
                    alpha_threshold = 254
                    valid_mask = valid_mask & (alpha_channel > alpha_threshold)

                if np.any(valid_mask):
                    depth_min = float(depth[valid_mask].min())
                    depth_max = float(depth[valid_mask].max())

                    depth_16bit = np.full(depth.shape, 65535, dtype=np.uint16)

                    depth_norm = (depth[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8)
                    depth_16bit[valid_mask] = (depth_norm * 65535).astype(np.uint16)
                else:
                    depth_16bit = np.full(depth.shape, 65535, dtype=np.uint16)
                    depth_min = 0.0
                    depth_max = 1.0
            else:
                depth_16bit = np.full(depth.shape, 65535, dtype=np.uint16)
                depth_min = 0.0
                depth_max = 1.0

            depth_ranges.append((depth_min, depth_max))

            depth_path = renders_cond_dir / f"{i:03d}_depth.png"
            cv2.imwrite(str(depth_path), depth_16bit)

        frames = []
        for i in range(num_views):
            depth_min, depth_max = depth_ranges[i]

            K = Ks[i]
            img_width = images[i].size[0]
            focal_x = K[0, 0]
            camera_angle_x = 2 * np.arctan(img_width / (2 * focal_x))

            frame = {
                "file_path": f"{i:03d}.png",
                "camera_angle_x": float(camera_angle_x),
                "transform_matrix": c2ws[i].tolist(),
                "depth": {"min": depth_min, "max": depth_max},
            }
            frames.append(frame)

        transforms = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "scale": float(scale),
            "offset": offset.tolist(),
            "frames": frames,
        }

        transforms_path = renders_cond_dir / "transforms.json"
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=4)

        annotation_dir = Path(affogato_base_dir) / "annotations" / file_identifier
        xyzc_path = annotation_dir / "xyzc.npy"
        queries_path = annotation_dir / "queries.json"

        assert xyzc_path.exists(), f"Affordance annotations not found: {xyzc_path}"
        assert queries_path.exists(), f"Affordance queries not found: {queries_path}"

        xyzc = np.load(xyzc_path)
        xyz_annotation = xyzc[:, :3]
        heatmaps_annotation = xyzc[:, 3:]

        xyz_scaled = xyz_annotation * (0.5 / 0.45)
        coord_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        xyz_transformed = (coord_transform @ xyz_scaled.T).T

        xyz_clamped = np.clip(xyz_transformed, -0.5 + 1e-6, 0.5 - 1e-6)
        voxel_indices = np.floor((xyz_clamped + 0.5) * 64).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, 63)

        voxel_keys = np.ravel_multi_index(voxel_indices.T, (64, 64, 64))
        _, unique_indices = np.unique(voxel_keys, return_index=True)

        voxel_indices_unique = voxel_indices[unique_indices]
        voxel_heatmaps = heatmaps_annotation[unique_indices]

        with open(queries_path) as f:
            queries_data = json.load(f)
        queries = queries_data[0]["queries"]

        affordances_dir = output_dir / "affordances" / sha256
        affordances_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            affordances_dir / "affordance.npz",
            coords=voxel_indices_unique.astype(np.int32),
            heatmap=voxel_heatmaps.astype(np.float32),
            query=np.array(queries, dtype=object),
        )

        num_voxels = len(voxel_indices_unique)

        return {
            "sha256": sha256,
            "cond_rendered": True,
            "affordance_processed": True,
            "num_voxels": num_voxels,
        }

    except Exception as e:
        print(f"Error processing {file_identifier} ({sha256}): {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the processed data"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip already processed samples"
    )
    dataset_utils.add_args(parser)
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of this process in distributed processing"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of processes in distributed processing",
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate voxels, affordance.npz, and mesh.ply (overwrite existing files)",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Only process test split samples",
    )
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, "renders_cond"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "affordances"), exist_ok=True)

    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        print("Metadata not found. Generating...")
        metadata = dataset_utils.get_metadata(**opt)
        metadata.to_csv(os.path.join(opt.output_dir, "metadata.csv"), index=False)
    else:
        print("Loading existing metadata...")
        metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

    if opt.test_only:
        if "split" not in metadata.columns:
            print("Error: --test_only requires 'split' column in metadata.csv")
            sys.exit(1)
        before_count = len(metadata)
        metadata = metadata[metadata["split"] == "test"]
        print(f"Filtering test split: {len(metadata)} / {before_count} samples")

    if opt.world_size > 1:
        metadata = metadata.iloc[opt.rank :: opt.world_size]
        print(f"Rank {opt.rank}/{opt.world_size}: Processing {len(metadata)} samples")

    if opt.skip_existing and not opt.force:

        def is_processed(row):
            sha256 = row["sha256"]
            transforms_exist = os.path.exists(
                os.path.join(opt.output_dir, "renders_cond", sha256, "transforms.json")
            )
            voxels_exist = os.path.exists(os.path.join(opt.output_dir, "voxels", f"{sha256}.ply"))
            affordance_exist = os.path.exists(
                os.path.join(opt.output_dir, "affordances", sha256, "affordance.npz")
            )
            return transforms_exist and voxels_exist and affordance_exist

        before_count = len(metadata)
        metadata = metadata[~metadata.apply(is_processed, axis=1)]
        print(f"Skipping {before_count - len(metadata)} already processed samples")

    print(f"Processing {len(metadata)} samples...")
    if opt.force:
        print("Force mode: Overwriting existing voxels, affordance.npz, and mesh.ply")

    timestamp = str(int(time.time()))

    from functools import partial

    process_func = partial(
        process_sample, affogato_base_dir=opt.affogato_base_dir, force=opt.force
    )

    records_df = dataset_utils.foreach_instance(
        metadata,
        opt.output_dir,
        process_func,
        max_workers=opt.max_workers,
        desc=f"Processing Affogato (rank {opt.rank})",
        affogato_base_dir=opt.affogato_base_dir,
    )

    if len(records_df) > 0:
        output_file = os.path.join(opt.output_dir, f"cond_rendered_{timestamp}_{opt.rank}.csv")
        records_df.to_csv(output_file, index=False)
        print(f"Saved processing records to {output_file}")
        print(f"Successfully processed: {len(records_df)} samples")
    else:
        print("No samples were successfully processed")

    print("\nProcessing complete!")
    print("To update metadata, run:")
    print("  python dataset_toolkits/build_metadata.py Affogato \\")
    print(f"    --output_dir {opt.output_dir} \\")
    print(f"    --affogato_base_dir {opt.affogato_base_dir} \\")
    print(f"    --trellis_data_dir {opt.get('trellis_data_dir', '~/datasets/trellis')} \\")
    print("    --from_file --field cond_rendered")
