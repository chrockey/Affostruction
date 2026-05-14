"""Affordance-driven active view selection (Sec. 3.4 of the paper).

Given a reconstruction's sparse voxels + a per-voxel affordance heatmap and
a dataset-style ``transforms.json`` listing the K candidate camera poses,
score each pose by ``S(π) = Σ pixel intensities`` of the rendered heatmap
and return the highest-scoring pose.

Coordinate conventions (mirrors ``examples/active_holistic_affordance.py``
on the ``main`` branch):

- World frame is Blender/NeRF: +Z up, +Y forward, +X right. The
  reconstruction lives in this frame after ``_depth_to_voxel_coords``
  flips the dataset's c2w cols 1,2.
- Cameras are loaded as Blender c2w from ``transforms.json`` and
  converted to OpenCV w2c (+Z forward, +Y down) via
  ``c2w_corrected = c2w.copy(); c2w_corrected[:3, 1:3] *= -1;
  w2c = inv(c2w_corrected)``.
- Intrinsics are NORMALIZED (focal = 0.5 / tan(fov/2), cx=cy=0.5);
  pixel space is recovered via ``u_px = u_norm * W``.

Two backends:

- ``mesh`` (paper version): decode the SLAT latent into a mesh, paint
  per-vertex affordance probability (looked up from the dense voxel
  grid), then rasterize per candidate pose using the nvdiffrast
  ``MeshRenderer``. Requires ``nvdiffrast``.
- ``voxel`` (no mesh decode): per-pixel raycast through a dense
  occupancy/probability volume. Each ray returns the probability of
  the first occupied voxel hit, giving exact self-occlusion without
  any mesh decoding. Decoder-free.
"""

import json
from typing import List, Tuple

import numpy as np
import torch


def load_poses_from_transforms(
    path: str, device: str = "cuda"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load a dataset ``transforms.json`` and return ``(extrinsics, intrinsics)``.

    Each ``frames[i].transform_matrix`` is a Blender c2w; we flip cols 1,2
    and invert to get the OpenCV w2c expected by the renderer. Intrinsics
    are normalized from ``camera_angle_x``.
    """
    with open(path) as f:
        d = json.load(f)
    extr_list: List[torch.Tensor] = []
    intr_list: List[torch.Tensor] = []
    for fr in d["frames"]:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w_corrected = c2w.copy()
        c2w_corrected[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w_corrected)
        extr_list.append(torch.from_numpy(w2c).float().to(device))
        fov = float(fr["camera_angle_x"])
        focal = 0.5 / np.tan(fov / 2.0)
        intr_list.append(
            torch.tensor(
                [[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
        )
    return extr_list, intr_list


def voxel_probs_to_vertex_heatmap(
    vertices: torch.Tensor,
    coords: torch.Tensor,
    probs: torch.Tensor,
    voxel_resolution: int,
) -> torch.Tensor:
    """Look up the per-vertex affordance probability from the dense voxel grid.

    Args:
        vertices: (V, 3) mesh vertices in the canonical cube ``[-0.5, 0.5]``.
        coords:   (N, 4) sparse voxel coords ``[batch, x, y, z]``.
        probs:    (N,) per-voxel probability.
        voxel_resolution: grid resolution (e.g. 64).
    """
    device = vertices.device
    R = int(voxel_resolution)
    dense = torch.zeros(R, R, R, device=device, dtype=probs.dtype)
    cx = coords[:, 1].long().to(device).clamp(0, R - 1)
    cy = coords[:, 2].long().to(device).clamp(0, R - 1)
    cz = coords[:, 3].long().to(device).clamp(0, R - 1)
    dense[cx, cy, cz] = probs.to(dense.dtype).to(device)

    vi = ((vertices.float() + 0.5) * R).floor().long().clamp(0, R - 1)
    return dense[vi[:, 0], vi[:, 1], vi[:, 2]]


class ViewSelectionPipeline:
    """Active view selection driven by predicted affordance heatmaps.

    Candidate camera poses always come from a dataset-style ``transforms.json``;
    the bundled samples ship a 40-frame ``candidate_transforms.json`` copied
    from the Affogato render set.
    """

    DEFAULT_RESOLUTION = 512

    def __init__(self, resolution: int = DEFAULT_RESOLUTION):
        self.resolution = resolution
        # Lazily cached nvdiffrast MeshRenderer. Built once and reused
        # across mesh-mode calls so the dr.RasterizeCudaContext (and the
        # CUDA setup it triggers) isn't paid per render. Keyed by
        # (resolution,) so changing self.resolution rebuilds the cache.
        self._mesh_renderer = None
        self._mesh_renderer_key = None

    def cuda(self) -> "ViewSelectionPipeline":
        return self

    def cpu(self) -> "ViewSelectionPipeline":
        return self

    def candidate_extrinsics_intrinsics(self, transforms_path: str):
        """Return ``(extrinsics_list, intrinsics_list)``.

        Each entry is a ``(4, 4)`` OpenCV w2c / ``(3, 3)`` normalized
        intrinsic, loaded from ``transforms_path`` via the Blender→OpenCV
        conversion described in the module docstring.
        """
        return load_poses_from_transforms(transforms_path)

    # ------------------------------------------------------------------ mesh

    @torch.no_grad()
    def score_mesh(
        self,
        mesh,
        coords: torch.Tensor,
        probs: torch.Tensor,
        voxel_resolution: int,
        *,
        transforms_path: str,
    ) -> torch.Tensor:
        """Paper version. Paint the mesh with the affordance heatmap,
        rasterize per candidate pose via nvdiffrast, return ``(K,)``
        Σ-pixel-intensity scores.
        """
        # Imported here so voxel-mode callers don't pay the nvdiffrast import.
        from ..utils.render_utils import get_renderer  # noqa: E402

        extr_list, intr_list = self.candidate_extrinsics_intrinsics(transforms_path)
        K = len(extr_list)
        if not getattr(mesh, "success", True):
            return torch.zeros(K)

        vertex_h = voxel_probs_to_vertex_heatmap(
            mesh.vertices, coords, probs, voxel_resolution
        )
        attrs = mesh.vertex_attrs
        if attrs is None:
            attrs = torch.zeros(
                mesh.vertices.shape[0], 6, device=mesh.vertices.device
            )
        else:
            attrs = attrs.clone()
        attrs[:, :3] = vertex_h.unsqueeze(-1).expand(-1, 3)
        mesh.vertex_attrs = attrs

        if self._mesh_renderer is None or self._mesh_renderer_key != (self.resolution,):
            self._mesh_renderer = get_renderer(
                mesh, resolution=self.resolution, near=1.0, far=100.0, ssaa=1
            )
            self._mesh_renderer_key = (self.resolution,)
        renderer = self._mesh_renderer

        # MeshRenderer supports batched extrinsics/intrinsics; nvdiffrast
        # rasterizes all candidate views in a single CUDA call.
        extr_batch = torch.stack([e.to(mesh.vertices.device) for e in extr_list])
        intr_batch = torch.stack([k.to(mesh.vertices.device) for k in intr_list])
        res = renderer.render(mesh, extr_batch, intr_batch, return_types=["color"])
        color = res["color"]
        # Batched output shape is (B, H, W, 3) per the MeshRenderer impl.
        if color.dim() == 4 and color.shape[-1] == 3:
            return color[..., 0].flatten(1).sum(dim=1).cpu()
        # Fallback for unexpected layouts.
        return color.reshape(K, -1).sum(dim=1).cpu()

    # ----------------------------------------------------------------- voxel

    @torch.no_grad()
    def score_voxels(
        self,
        coords: torch.Tensor,
        probs: torch.Tensor,
        voxel_resolution: int,
        *,
        transforms_path: str,
        ray_samples_per_voxel: float = 1.0,
        ray_near: float = 0.4,
        ray_far: float = 3.6,
    ) -> torch.Tensor:
        """Per-pixel raycast through a dense occupancy/probability volume.

        For each candidate view, cast one ray per pixel from the camera
        through the canonical cube, sample the dense voxel grid along the
        ray, and accumulate the probability of the **first occupied voxel
        hit**. Self-occlusion is exact: back voxels are skipped because
        the closer one terminates the ray. Returns ``(K,)`` scores
        (Σ pixel intensity per view).

        Args:
            ray_samples_per_voxel: depth samples per cube-side voxel
                length along each ray. ``1`` ≈ one sample per voxel.
            ray_near, ray_far: depth range to sample in camera frame.
        """
        device = probs.device
        extr_list, intr_list = self.candidate_extrinsics_intrinsics(transforms_path)
        K = len(extr_list)
        R = int(voxel_resolution)
        # Render at the voxel-grid resolution so each pixel covers roughly
        # one projected voxel. There is no surface information beyond the
        # sparse-structure decoder's voxel grid, so a higher render
        # resolution would just smear each voxel across multiple pixels
        # without adding detail.
        H = W = R

        # Build dense affordance + occupancy volumes (R, R, R).
        dense_p = torch.zeros(R, R, R, device=device, dtype=probs.dtype)
        cx_v = coords[:, 1].long().to(device).clamp(0, R - 1)
        cy_v = coords[:, 2].long().to(device).clamp(0, R - 1)
        cz_v = coords[:, 3].long().to(device).clamp(0, R - 1)
        dense_p[cx_v, cy_v, cz_v] = probs.to(dense_p.dtype).to(device)
        occ = torch.zeros(R, R, R, device=device)
        occ[cx_v, cy_v, cz_v] = 1.0

        # Pixel grid (center of each pixel, normalized [0, 1] coords).
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
            indexing="ij",
        )
        u_norm = (xx + 0.5) / W
        v_norm = (yy + 0.5) / H

        # Number of depth samples so that step ≤ 1 / (R * samples_per_voxel)
        # in world units across the [near, far] range. Without this, the
        # ray skips alternate voxel slices and produces speckled hollows
        # where occupied voxels live between adjacent samples.
        depth_span = max(ray_far - ray_near, 1e-3)
        T = int(np.ceil(depth_span * R * ray_samples_per_voxel)) + 1
        t_vals = torch.linspace(ray_near, ray_far, T, device=device)

        # Batched across all K candidate views. Stack extrinsics/intrinsics
        # so all rays for all views are sampled in one fused tensor pass.
        extr_batch = torch.stack([e.to(device).float() for e in extr_list])  # (K,4,4)
        intr_batch = torch.stack([k.to(device).float() for k in intr_list])  # (K,3,3)
        fx = intr_batch[:, 0, 0].view(K, 1, 1)
        fy = intr_batch[:, 1, 1].view(K, 1, 1)
        cxn = intr_batch[:, 0, 2].view(K, 1, 1)
        cyn = intr_batch[:, 1, 2].view(K, 1, 1)
        # Ray direction in camera frame (OpenCV: +z forward). (K, H, W, 3)
        d_cam_x = (u_norm.unsqueeze(0) - cxn) / fx
        d_cam_y = (v_norm.unsqueeze(0) - cyn) / fy
        d_cam_z = torch.ones_like(d_cam_x)
        d_cam = torch.stack([d_cam_x, d_cam_y, d_cam_z], dim=-1)
        d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True)
        # Camera->world rotation = R_w2c^T. d_world = R_w2c^T @ d_cam.
        R_w2c = extr_batch[:, :3, :3]
        t_w2c = extr_batch[:, :3, 3]
        cam_center = -torch.bmm(R_w2c.transpose(1, 2), t_w2c.unsqueeze(-1)).squeeze(-1)
        d_world = torch.einsum("kij,khwj->khwi", R_w2c.transpose(1, 2), d_cam)

        # Sample world points along each ray: (K, H, W, T, 3).
        pts = cam_center.view(K, 1, 1, 1, 3) + t_vals.view(1, 1, 1, T, 1) * d_world.unsqueeze(3)
        idx_f = (pts + 0.5) * R
        inb = (idx_f >= 0).all(dim=-1) & (idx_f < R).all(dim=-1)
        idx = idx_f.long().clamp(0, R - 1)
        occ_samples = occ[idx[..., 0], idx[..., 1], idx[..., 2]] * inb.float()
        p_samples = dense_p[idx[..., 0], idx[..., 1], idx[..., 2]] * inb.float()
        cumsum = occ_samples.cumsum(dim=-1)
        first_hit = (occ_samples > 0) & (cumsum == 1)
        return (p_samples * first_hit.float()).reshape(K, -1).sum(dim=1).cpu()

    # --------------------------------------------------------------- run

    @torch.no_grad()
    def run(
        self,
        coords: torch.Tensor,
        probs: torch.Tensor,
        voxel_resolution: int,
        *,
        transforms_path: str,
        mode: str = "voxel",
        mesh=None,
    ) -> dict:
        """Return the highest-scoring candidate pose.

        Args:
            coords: (N, 4) sparse voxel coords.
            probs:  (N,) affordance probability.
            voxel_resolution: voxel grid resolution.
            transforms_path: dataset ``transforms.json`` listing the K
                candidate poses (Blender c2w). Required.
            mode: ``"mesh"`` (paper, needs nvdiffrast + a decoded mesh) or
                ``"voxel"`` (skip mesh decoding).
            mesh: ``MeshExtractResult`` from the SLAT mesh decoder. Required
                when ``mode == "mesh"``.

        Returns dict with: ``index``, ``score``, ``scores`` (K,),
        ``extrinsics``, ``intrinsics``, ``origin_world``, ``yaw``, ``pitch``.
        """
        if mode == "mesh":
            if mesh is None:
                raise ValueError("mode='mesh' requires a decoded mesh.")
            scores = self.score_mesh(
                mesh, coords, probs, voxel_resolution, transforms_path=transforms_path
            )
        elif mode == "voxel":
            scores = self.score_voxels(
                coords, probs, voxel_resolution, transforms_path=transforms_path
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'mesh' or 'voxel'.")

        extr_list, intr_list = self.candidate_extrinsics_intrinsics(transforms_path)
        idx = int(scores.argmax().item())
        chosen_extr = extr_list[idx]
        # Recover camera origin in world frame from w2c: cam_origin = -R^T @ t.
        R = chosen_extr[:3, :3]
        t = chosen_extr[:3, 3]
        origin_world = (-R.T @ t).detach().cpu().numpy()
        r_w = float(np.linalg.norm(origin_world))
        yaw = float(np.arctan2(origin_world[1], origin_world[0])) if r_w > 1e-6 else 0.0
        pitch = float(np.arcsin(origin_world[2] / r_w)) if r_w > 1e-6 else 0.0
        return {
            "index": idx,
            "score": float(scores[idx]),
            "scores": scores,
            "extrinsics": chosen_extr,
            "intrinsics": intr_list[idx],
            "origin_world": origin_world,
            "yaw": yaw,
            "pitch": pitch,
            "mode": mode,
        }
