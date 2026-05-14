import torch
import nvdiffrast.torch as dr
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F


def intrinsics_to_projection(
    intrinsics: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = -2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.0
    return ret


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
    """

    def __init__(self, rendering_options={}, device="cuda"):
        self.rendering_options = edict({"resolution": None, "near": None, "far": None, "ssaa": 1})
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device = device

    def render(
        self,
        mesh: MeshExtractResult,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_types=["mask", "normal", "depth"],
    ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) or (B, 4, 4) camera extrinsics (batch supported)
            intrinsics (torch.Tensor): (3, 3) or (B, 3, 3) camera intrinsics (batch supported)
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] or [B, H, W, 3] rendered color image
                depth (torch.Tensor): [H, W] or [B, H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] or [B, H, W, 3] rendered normal image
                normal_map (torch.Tensor): [3, H, W] or [B, H, W, 3] rendered normal map image
                mask (torch.Tensor): [H, W] or [B, H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        # Handle batch dimensions
        if extrinsics.dim() == 2:
            # Single view: [4, 4] -> [1, 4, 4]
            extrinsics = extrinsics.unsqueeze(0)
            intrinsics = intrinsics.unsqueeze(0)
            batch_size = 1
            squeeze_output = True
        else:
            # Batch views: [B, 4, 4]
            batch_size = extrinsics.shape[0]
            squeeze_output = False

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros(
                (batch_size, resolution, resolution, 3), dtype=torch.float32, device=self.device
            )
            ret_dict = {
                k: default_img if k in ["normal", "normal_map", "color"] else default_img[..., :1]
                for k in return_types
            }
            if squeeze_output:
                ret_dict = {
                    k: v.squeeze(0)
                    if k not in ["normal", "normal_map", "color"]
                    else v.permute(0, 3, 1, 2).squeeze(0)
                    for k, v in ret_dict.items()
                }
            return ret_dict

        # Compute perspectives for each batch
        perspectives = torch.stack(
            [intrinsics_to_projection(intr, near, far) for intr in intrinsics]
        )

        RT = extrinsics
        full_proj = torch.bmm(perspectives, extrinsics)

        # Expand vertices for batch
        vertices = mesh.vertices.unsqueeze(0).expand(batch_size, -1, -1)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, _ = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa)
        )

        out_dict = edict()
        for type in return_types:
            img = None
            if type == "mask":
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal":
                # Expand face_normal for batch
                # face_normal shape: [N_faces, 3, 3] -> reshape to [batch_size, N_faces*3, 3]
                face_normal_batch = mesh.face_normal.reshape(1, -1, 3).expand(batch_size, -1, -1)
                img = dr.interpolate(
                    face_normal_batch,
                    rast,
                    torch.arange(
                        mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int
                    ).reshape(-1, 3),
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # normalize norm pictures
                img = (img + 1) / 2
            elif type == "normal_map":
                # Expand vertex_attrs for batch
                vertex_attrs_batch = (
                    mesh.vertex_attrs[:, 3:].unsqueeze(0).expand(batch_size, -1, -1)
                )
                img = dr.interpolate(vertex_attrs_batch.contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color":
                # Expand vertex_attrs for batch
                vertex_attrs_batch = (
                    mesh.vertex_attrs[:, :3].unsqueeze(0).expand(batch_size, -1, -1)
                )
                img = dr.interpolate(vertex_attrs_batch.contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)

            if ssaa > 1:
                img = F.interpolate(
                    img.permute(0, 3, 1, 2),
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                if squeeze_output:
                    img = img.squeeze(0)
                else:
                    img = img.permute(0, 2, 3, 1).squeeze(-1) if img.shape[1] == 1 else img
            else:
                if squeeze_output:
                    img = img.permute(0, 3, 1, 2).squeeze(0)
                else:
                    # Keep batch dimension: [B, H, W, C] -> [B, H, W] or [B, H, W, C]
                    if img.shape[-1] == 1:
                        img = img.squeeze(-1)  # [B, H, W]
                    # else keep [B, H, W, 3]
            out_dict[type] = img

        return out_dict
