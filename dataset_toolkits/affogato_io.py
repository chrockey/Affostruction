"""Readers for the raw Affogato bundles: meshes, EXR depth and cameras."""

import json
import os
import re
from typing import Tuple

import cv2
import numpy as np
import trimesh

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_mesh_trimesh(mesh_path):
    """
    Load a 3D mesh from a GLB file using trimesh.

    Args:
        glb_path (str): Path to the GLB file

    Returns:
        trimesh.Trimesh or trimesh.Scene: The loaded mesh or scene
    """
    try:
        mesh = trimesh.load_mesh(mesh_path, file_type="glb")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        return mesh
    except Exception as e:
        print(f"Error loading mesh with trimesh: {e}")
        return None


def read_depth(exr_path_or_bytes):
    normal, depth = read_exr(exr_path_or_bytes)
    return depth


def read_cam_params(json_file_or_bytes, image_wh: Tuple[int, int]):
    """
    Read camera parameters from a JSON file.

    Args:
        json_file (str): Path to the JSON file
        image_wh (tuple): Shape of the image (height, width)

    Returns:
        tuple: Camera pose (c2w) and intrinsic matrix (K)
    """
    if isinstance(json_file_or_bytes, str):
        with open(json_file_or_bytes, "r", encoding="utf8") as reader:
            json_content = json.load(reader)
    else:
        json_content = json.load(json_file_or_bytes)

    c2w = np.eye(4)
    c2w[:3, 0] = np.array(json_content["x"])
    c2w[:3, 1] = np.array(json_content["y"])
    c2w[:3, 2] = np.array(json_content["z"])
    c2w[:3, 3] = np.array(json_content["origin"])
    swap_flip = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=c2w.dtype,
    )
    c2w_transformed = swap_flip @ c2w

    fov = json_content["x_fov"]
    fx = image_wh[0] / 2 / np.tan(fov / 2)
    fy = image_wh[1] / 2 / np.tan(fov / 2)
    K = np.array(
        [
            [fx, 0, (image_wh[0]) / 2],
            [0, fy, (image_wh[1]) / 2],
            [0, 0, 1],
        ]
    )

    return c2w_transformed, K


def read_exr(exr_path_or_bytes):
    """
    Load a normal and depth map from an EXR file.

    Args:
        exr_path (str): Path to the EXR file

    Returns:
        tuple: Normal and depth map
    """
    if isinstance(exr_path_or_bytes, str):
        normal_d = cv2.imread(exr_path_or_bytes, cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )
    elif hasattr(exr_path_or_bytes, "read"):
        image_bytes = exr_path_or_bytes.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        normal_d = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )
    else:
        raise ValueError("Invalid input type for load_exr")

    normal, depth = normal_d[..., :3], normal_d[..., 3]
    return normal, depth
