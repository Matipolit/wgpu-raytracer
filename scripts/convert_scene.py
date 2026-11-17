"""Convert an OBJ/FBX scene to a self-contained GLB using Blender.

Usage: blender --background --factory-startup --python scripts/convert_scene.py -- IN OUT.glb [SCALE]

SCALE wraps the scene in a uniformly scaled root node (e.g. 0.01 for centimetres),
keeping it inside the tracer's fixed [0.001, 1000] ray interval.
"""
import sys

import bpy

arguments = sys.argv[sys.argv.index("--") + 1 :]
source, target = arguments[:2]
scale = float(arguments[2]) if len(arguments) > 2 else 1.0
bpy.ops.wm.read_factory_settings(use_empty=True)
if source.lower().endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=source)
elif source.lower().endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=source)
else:
    bpy.ops.import_scene.gltf(filepath=source)
bpy.ops.export_scene.gltf(
    filepath=target,
    export_format="GLB",
    export_tangents=False,
    export_animations=False,
    export_cameras=False,
    export_lights=False,
    export_draco_mesh_compression_enable=False,
    export_image_format="AUTO",
)

# The renderer supports alpha cutouts but not blended transparency: export
# Blender's alpha-blended foliage/decals as MASK materials (cutoff 0.5).
import json
import struct

with open(target, "rb") as f:
    data = bytearray(f.read())
length = struct.unpack_from("<I", data, 12)[0]
document = json.loads(data[20 : 20 + length])
for material in document.get("materials", []):
    if material.get("alphaMode") == "BLEND":
        material["alphaMode"] = "MASK"
        material.setdefault("alphaCutoff", 0.5)
if scale != 1.0:
    scene = document["scenes"][document.get("scene", 0)]
    document["nodes"].append({"name": "scale", "scale": [scale] * 3, "children": scene["nodes"]})
    scene["nodes"] = [len(document["nodes"]) - 1]
encoded = json.dumps(document, separators=(",", ":")).encode()
encoded += b" " * (-len(encoded) % 4)
rest = data[20 + length :]
out = bytearray(data[:12]) + struct.pack("<I", len(encoded)) + b"JSON" + encoded + rest
struct.pack_into("<I", out, 8, len(out))
with open(target, "wb") as f:
    f.write(out)
