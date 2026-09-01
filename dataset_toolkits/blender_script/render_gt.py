import argparse, sys, os, math, re
from typing import *
import bpy
from mathutils import Vector
import numpy as np
import json


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    "PNG": "png",
    "JPEG": "jpg",
    "OPEN_EXR": "exr",
    "TIFF": "tiff",
    "BMP": "bmp",
    "HDR": "hdr",
    "TARGA": "tga",
}


def init_render(engine="CYCLES", resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGB"
    bpy.context.scene.render.film_transparent = False

    if not bpy.data.worlds:
        bpy.ops.world.new()
    world = bpy.data.worlds[0]
    if world.use_nodes:
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (0, 0, 0, 1)
            bg_node.inputs[1].default_value = 0
    else:
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
        world.node_tree.nodes["Background"].inputs[1].default_value = 0

    if engine == "CYCLES":
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
        bpy.context.scene.cycles.filter_type = "BOX"
        bpy.context.scene.cycles.filter_width = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
        bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
        bpy.context.scene.cycles.use_denoising = False

        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    else:
        bpy.context.scene.eevee.taa_render_samples = 4
        bpy.context.scene.eevee.gi_diffuse_bounces = 0


def init_nodes(save_normal=False):
    """
    Initialize compositor nodes.
    Note: Normal rendering now uses material override instead of compositor nodes
    for accurate world-space normal output.
    """
    return {}, {}


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    from mathutils import Vector

    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    return cam


def init_lighting():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    default_light = bpy.data.objects.new(
        "Default_Light", bpy.data.lights.new("Default_Light", type="POINT")
    )
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)

    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)

    bottom_light = bpy.data.objects.new(
        "Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA")
    )
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)

    return {"default_light": default_light, "top_light": top_light, "bottom_light": bottom_light}


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
    else:
        import_function(filepath=object_path)


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [
        obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)
    ]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def main(arg):
    import sys

    os.makedirs(arg.output_folder, exist_ok=True)

    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=False)
    outputs, spec_nodes = init_nodes(save_normal=True)

    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()

    mesh_objects = [obj for obj in bpy.context.scene.objects.values() if obj.type == "MESH"]
    if len(mesh_objects) == 0:
        print("[ERROR] No mesh objects found in scene!")
        return
    print(f"[INFO] Scene initialized with {len(mesh_objects)} mesh object(s)")

    scale, offset = normalize_scene()
    bpy.context.view_layer.update()
    print(f"[INFO] Scene normalized (scale={scale:.4f})")

    cam = init_camera()
    init_lighting()
    print("[INFO] Camera and lighting initialized.")

    view = json.loads(arg.view)

    cam.location = (
        view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
        view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
        view["radius"] * np.sin(view["pitch"]),
    )

    from mathutils import Vector

    direction = Vector((0, 0, 0)) - Vector(cam.location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    cam.data.lens = 16 / np.tan(view["fov"] / 2)

    bpy.context.view_layer.update()

    print(f"[INFO] Rendering view at yaw={view['yaw']:.4f}, pitch={view['pitch']:.4f}")

    print("[INFO] Rendering RGB...")
    bpy.context.scene.render.filepath = os.path.join(arg.output_folder, "rgb.png")
    view_layer_name = (
        "View Layer" if "View Layer" in bpy.context.scene.view_layers else "ViewLayer"
    )
    bpy.context.scene.view_layers[view_layer_name].material_override = None
    bpy.ops.render.render(write_still=True)
    bpy.context.view_layer.update()

    with open(os.path.join(arg.output_folder, "camera.txt"), "w") as f:
        f.write(f"{view['yaw']} {view['pitch']}\n")

    print(f"[INFO] RGB rendering complete. Output saved to {arg.output_folder}")
    print(f"[INFO] Normal will be rendered separately using TRELLIS renderer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renders GT mesh with random camera view for reconstruction evaluation."
    )
    parser.add_argument(
        "--view",
        type=str,
        help="JSON string of single view. Contains {yaw, pitch, radius, fov} object.",
    )
    parser.add_argument("--object", type=str, help="Path to the 3D model file to be rendered.")
    parser.add_argument(
        "--output_folder", type=str, default="/tmp", help="The path the output will be dumped to."
    )
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the images.")
    parser.add_argument(
        "--engine",
        type=str,
        default="CYCLES",
        help="Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...",
    )
    parser.add_argument(
        "--split_normal", action="store_true", help="Split the normals of the mesh."
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    main(args)
