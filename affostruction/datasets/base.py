from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
        split (str, optional): split column value to keep ("train", "val", "test").
            If None, uses all data.
        manifest (str, optional): path to an object list (one ``sha256`` or
            ``sha256,uid`` per line). Takes precedence over ``split`` and is how
            the evaluation splits are pinned down.
    """

    def __init__(
        self,
        roots: str,
        split: Optional[Literal["train", "val", "test"]] = None,
        manifest: Optional[str] = None,
    ):
        super().__init__()
        self.roots = roots.split(",")
        self.split = split
        self.manifest = manifest
        selected = None
        if manifest is not None:
            with open(manifest) as f:
                selected = {line.split(",")[0].strip() for line in f if line.strip()}
        self.instances = []
        self.metadata = pd.DataFrame()

        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, "metadata.csv"))
            self._stats[key]["Total"] = len(metadata)

            if selected is not None:
                metadata = metadata[metadata["sha256"].isin(selected)]
                self._stats[key][f"Manifest={os.path.basename(manifest)}"] = len(metadata)
            elif self.split is not None and "split" in metadata.columns:
                metadata = metadata[metadata["split"] == self.split]
                self._stats[key][f"Split={self.split}"] = len(metadata)

            metadata, stats = self.filter_metadata(metadata, root)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata["sha256"].values])
            metadata.set_index("sha256", inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

    @abstractmethod
    def filter_metadata(
        self, metadata: pd.DataFrame, root: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass

    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        max_retries = 100
        for _ in range(max_retries):
            try:
                root, instance = self.instances[index]
                return self.get_instance(root, instance)
            except Exception as e:
                print(f"[dataset] failed to load {self.instances[index]}: {e!r} — retrying with a random sample")
                index = np.random.randint(0, len(self))
        raise RuntimeError(f"Failed to load a sample after {max_retries} attempts — check the dataset.")

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f"  - Total instances: {len(self)}")
        lines.append(f"  - Sources:")
        for key, stats in self._stats.items():
            lines.append(f"    - {key}:")
            for k, v in stats.items():
                lines.append(f"      - {k}: {v}")
        return "\n".join(lines)


class MultiViewVoxelConditioningMixin:
    """
    Multi-view DINOv2 sparse voxel conditioning mixin.
    Samples multiple views and returns voxel data as a list for later aggregation.

    For training: randomly samples 1~max_views views per instance.
    For val/test: uses fixed num_views (default=1) for consistent evaluation.
    """

    def __init__(
        self,
        roots,
        *,
        voxel_resolution: int = 16,
        image_size: int = 224,
        max_views: int = 16,
        num_views: Optional[int] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        **kwargs,
    ):
        self.voxel_resolution = voxel_resolution
        self.image_size = image_size
        self.max_views = max_views
        self.num_views = (
            num_views if num_views is not None else (1 if split in ["val", "test"] else None)
        )
        super().__init__(roots, split=split, **kwargs)

    def filter_metadata(self, metadata, root):
        result = super().filter_metadata(metadata, root)
        if result is not None:
            metadata, stats = result
        else:
            stats = {}
        metadata = metadata[metadata["cond_rendered"]]
        stats["Cond rendered"] = len(metadata)
        return metadata, stats

    def _depth_to_voxel_coords(
        self,
        depth_array,
        alpha_mask,
        transform_matrix,
        depth_min,
        depth_max,
        camera_angle_x,
        height,
        width,
    ):
        """Convert depth image to voxel coordinates and UV coordinates.

        The 16-bit depth PNGs fill background pixels with the sentinel max
        (normalised ~1.0), and the alpha channel can be a pixel wider than the
        rendered object, so background-depth pixels are rejected explicitly on
        top of the alpha test.

        Objects are normalised into [-0.5, 0.5]^3, so surfaces flush against a
        face sit exactly on the boundary and depth quantisation scatters them a
        hair outside; those are clamped into the edge voxel, while anything
        further out than one voxel is treated as broken input and dropped.
        """
        absolute_depth = depth_array * (depth_max - depth_min) + depth_min

        DEPTH_BG_NORMALIZED = 0.999
        valid_mask = (absolute_depth > 0) & (alpha_mask > 0) & (depth_array < DEPTH_BG_NORMALIZED)
        if not valid_mask.any():
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        focal_length = width / (2.0 * np.tan(camera_angle_x / 2.0))
        cx, cy = width / 2.0, height / 2.0

        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x_coords = x_coords + 0.5
        y_coords = y_coords + 0.5

        x_cam = (x_coords - cx) * absolute_depth / focal_length
        y_cam = (y_coords - cy) * absolute_depth / focal_length
        z_cam = absolute_depth

        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        xyz_cam_valid = xyz_cam[valid_mask]

        c2w = np.array(transform_matrix)
        c2w_corrected = c2w.copy()
        c2w_corrected[:3, 1:3] *= -1

        xyz_cam_homo = np.concatenate(
            [xyz_cam_valid, np.ones((xyz_cam_valid.shape[0], 1))], axis=1
        )
        xyz_world_homo = xyz_cam_homo @ c2w_corrected.T
        xyz_world = xyz_world_homo[:, :3]

        tolerance = 1.0 / self.voxel_resolution
        bounds_mask = (xyz_world >= -0.5 - tolerance) & (xyz_world <= 0.5 + tolerance)
        bounds_mask = bounds_mask.all(axis=1)
        xyz_world = xyz_world[bounds_mask]

        if xyz_world.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

        voxel_coords = ((xyz_world + 0.5) * self.voxel_resolution).astype(np.int64)
        voxel_coords = np.clip(voxel_coords, 0, self.voxel_resolution - 1)

        valid_indices = np.argwhere(valid_mask)
        uv_coords = valid_indices[:, [1, 0]].astype(np.float32)

        uv_coords = uv_coords[bounds_mask]

        uv_coords[:, 0] = uv_coords[:, 0] / width * 2 - 1
        uv_coords[:, 1] = uv_coords[:, 1] / height * 2 - 1

        R = self.voxel_resolution
        voxel_codes = voxel_coords[:, 0] * R * R + voxel_coords[:, 1] * R + voxel_coords[:, 2]

        unique_codes, inverse_indices = np.unique(voxel_codes, return_inverse=True)

        num_unique = unique_codes.shape[0]
        unique_uv_coords = np.zeros((num_unique, 2), dtype=np.float32)

        for i in range(num_unique):
            mask = inverse_indices == i
            unique_uv_coords[i] = uv_coords[mask].mean(axis=0)

        unique_voxel_coords = np.zeros((num_unique, 3), dtype=np.int64)
        unique_voxel_coords[:, 0] = unique_codes // (R * R)
        unique_voxel_coords[:, 1] = (unique_codes % (R * R)) // R
        unique_voxel_coords[:, 2] = unique_codes % R

        return unique_voxel_coords, unique_uv_coords

    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        renders_dir = os.path.join(root, "renders_cond", instance)
        with open(os.path.join(renders_dir, "transforms.json")) as f:
            metadata = json.load(f)

        n_views_total = len(metadata["frames"])

        if self.split in ["test", "val"]:
            actual_num_views = self.num_views
            seed = int(instance[:16], 16) % (2**32)
            rng = np.random.RandomState(seed)
            view_indices = rng.choice(n_views_total, size=actual_num_views, replace=False)
        else:
            actual_num_views = np.random.randint(1, self.max_views + 1)
            view_indices = np.random.choice(n_views_total, size=actual_num_views, replace=False)

        views_data = []
        for view_idx in view_indices:
            frame = metadata["frames"][view_idx]

            rgb_filename = os.path.basename(frame["file_path"])
            view_name = rgb_filename.split(".")[0]
            rgb_path = os.path.join(renders_dir, rgb_filename)
            depth_path = os.path.join(renders_dir, f"{view_name}_depth.png")

            rgb_image_orig = Image.open(rgb_path).convert("RGBA")
            rgb_image = rgb_image_orig.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
            rgb_array = np.array(rgb_image).astype(np.float32) / 255
            rgb_array = rgb_array[:, :, :3] * rgb_array[:, :, 3:]
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()

            alpha = np.array(rgb_image_orig)[:, :, 3].astype(np.float32) / 255

            depth_image = Image.open(depth_path)
            depth_array = np.array(depth_image).astype(np.float32) / 65535.0

            depth_image_resized = Image.fromarray(depth_array).resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
            depth_array_resized = np.array(depth_image_resized).astype(np.float32)

            alpha_resized = np.array(
                Image.fromarray(alpha).resize(
                    (self.image_size, self.image_size), Image.Resampling.LANCZOS
                )
            )

            voxel_coords, uv_coords = self._depth_to_voxel_coords(
                depth_array_resized,
                alpha_resized,
                frame["transform_matrix"],
                frame["depth"]["min"],
                frame["depth"]["max"],
                frame["camera_angle_x"],
                self.image_size,
                self.image_size,
            )

            depth_3ch = np.stack(
                [depth_array_resized, depth_array_resized, depth_array_resized], axis=0
            )
            depth_tensor = torch.from_numpy(depth_3ch).float()
            alpha_tensor = torch.from_numpy(alpha_resized).float()
            depth_tensor = depth_tensor * alpha_tensor.unsqueeze(0)

            views_data.append(
                {
                    "voxel_coords": torch.from_numpy(voxel_coords).long(),
                    "uv_coords": torch.from_numpy(uv_coords).float(),
                    "rgb": rgb_tensor,
                    "depth": depth_tensor,
                }
            )

        pack["cond"] = views_data
        return pack
