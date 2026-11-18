"""One of our scenes in Blender Cycles with the renderer's camera, sky and sun, for
checking the reference path tracer against an independent one.

    blender -b --python training/reference/cycles_render.py -- --model SCENE.glb
        --camera x,y,z,dx,dy,dz,fov --size WxH --sky SKY.json --bounces N
        --samples S --out OUT.f32 [--clay]

SKY.json is the renderer's `--print-sky` output: the atmosphere becomes an environment
texture evaluated with the same formula, the sun a sun lamp with the same irradiance
and angular size. `--bounces N` is the renderer's count (path vertices on surfaces,
each with a light sample): Cycles' max bounces is N - 1. No clamping, no adaptive
sampling, no denoising, box pixel filter. `--clay`: every surface a grey Lambertian
(the renderer's `--clay`). OUT.f32: linear rgb, f32, top row first.
"""
import argparse
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Vector

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--camera", required=True)
p.add_argument("--size", default="960x540")
p.add_argument("--sky", required=True)
p.add_argument("--bounces", type=int, default=2)
p.add_argument("--samples", type=int, default=1024)
p.add_argument("--out", required=True)
p.add_argument("--clay", action="store_true")
a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
width, height = map(int, a.size.split("x"))
sky = json.load(open(a.sky))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a.model)
scene = bpy.context.scene


def blender(v):
    """Renderer (y up) to Blender (z up), as the glTF importer converts."""
    return Vector((v[0], -v[2], v[1]))


# Camera.
c = [float(x) for x in a.camera.split(",")]
camera = bpy.data.cameras.new("camera")
camera.sensor_fit = "VERTICAL"
camera.angle_y = math.radians(c[6])
camera.clip_start, camera.clip_end = 0.001, 100000
holder = bpy.data.objects.new("camera", camera)
scene.collection.objects.link(holder)
holder.location = blender(c[:3])
holder.rotation_euler = blender(c[3:6]).normalized().to_track_quat("-Z", "Y").to_euler()
scene.camera = holder
scene.render.resolution_x, scene.render.resolution_y = width, height
scene.render.resolution_percentage = 100

# Atmosphere: the renderer's formula (raytracer.wgsl `radiance`, mirrored below the
# horizon) on an equirectangular image in Cycles' mapping.
w, h = 2048, 1024
u = (np.arange(w) + 0.5) / w
v = (np.arange(h) + 0.5) / h
theta = math.pi * (1 - v)[:, None]
phi = (math.pi - 2 * math.pi * u)[None, :]
bx, by, bz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * np.ones_like(phi)
d = np.stack([bx, bz, -by], -1)  # Blender direction -> renderer direction
sun = np.array(sky["sun_direction"])
gamma = np.arccos(np.clip(d @ sun, -1, 1))
cos_theta = np.abs(d[..., 1])
image = np.zeros((h, w, 4), dtype=np.float32)
image[..., 3] = 1
for channel in range(3):
    q = sky["params"][channel * 9 : channel * 9 + 9]
    cg = np.cos(gamma)
    mie = (1 + cg * cg) / np.power(1 + q[8] * q[8] - 2 * q[8] * cg, 1.5)
    left = 1 + q[0] * np.exp(q[1] / (cos_theta + 0.01))
    right = q[2] + q[3] * np.exp(q[4] * gamma) + q[5] * cg * cg + q[6] * mie + q[7] * np.sqrt(cos_theta)
    image[..., channel] = sky["radiances"][channel] * left * right
texture = bpy.data.images.new("sky", w, h, alpha=False, float_buffer=True)
texture.colorspace_settings.name = "Non-Color"
texture.pixels.foreach_set(image.ravel())
texture.pack()
world = bpy.data.worlds.new("world")
scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
nodes.clear()
environment = nodes.new("ShaderNodeTexEnvironment")
environment.image = texture
environment.interpolation = "Linear"
background = nodes.new("ShaderNodeBackground")
output = nodes.new("ShaderNodeOutputWorld")
world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
background.inputs["Strength"].default_value = 1

# Sun: irradiance = radiance x solid angle.
irradiance = np.array(sky["sun_radiance"]) * sky["sun_solid_angle"]
if irradiance.max() > 0:
    lamp = bpy.data.lights.new("sun", "SUN")
    lamp.angle = 2 * sky["sun_angular_radius"]
    lamp.energy = float(irradiance.max())
    lamp.color = tuple(float(x) for x in irradiance / irradiance.max())
    holder = bpy.data.objects.new("sun", lamp)
    scene.collection.objects.link(holder)
    holder.rotation_euler = (-blender(sun)).normalized().to_track_quat("-Z", "Y").to_euler()

# The renderer's per-scene material fixes (performance/scenes/material_overrides.txt).
import os

overrides = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../performance/scenes/material_overrides.txt")
if os.path.exists(overrides) and not a.clay:
    file = os.path.basename(a.model)
    rules = [line.split() for line in open(overrides) if line.strip() and not line.startswith("#")]
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        principled = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue

        def put(name, value):
            socket = principled.inputs.get(name)
            if socket is not None and not socket.is_linked:
                socket.default_value = value

        for rule in rules:
            if rule[0] != file or not (rule[1] == "*" or rule[1] in material.name):
                continue
            for setting in rule[2:]:
                key, value = setting.split("=")
                if key == "transmission":
                    put("Transmission Weight", float(value))
                elif key == "ior":
                    put("IOR", float(value))
                elif key == "specular":
                    put("Specular IOR Level", 0.5 * float(value))
                elif key == "roughness":
                    put("Roughness", float(value))
                elif key == "base":
                    put("Base Color", (*[float(x) for x in value.split(",")], 1))

if a.clay:
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    tree = clay.node_tree
    tree.nodes.clear()
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    diffuse.inputs["Roughness"].default_value = 0
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
    bpy.context.view_layer.material_override = clay

# Cycles: plain, unclamped path tracing.
scene.render.engine = "CYCLES"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "OPTIX"
preferences.refresh_devices()
for device in preferences.devices:
    device.use = device.type == "OPTIX"
cycles = scene.cycles
cycles.device = "GPU"
cycles.samples = a.samples
cycles.use_adaptive_sampling = False
cycles.use_denoising = False
indirect = a.bounces - 1
cycles.max_bounces = indirect
cycles.diffuse_bounces = indirect
cycles.glossy_bounces = indirect
cycles.transmission_bounces = indirect
cycles.volume_bounces = 0
cycles.transparent_max_bounces = 64
cycles.sample_clamp_direct = 0
cycles.sample_clamp_indirect = 0
cycles.blur_glossy = 0
cycles.caustics_reflective = True
cycles.caustics_refractive = True
cycles.light_sampling_threshold = 0
cycles.pixel_filter_type = "BOX"
cycles.film_exposure = 1
scene.render.film_transparent = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = a.out + ".exr"
bpy.ops.render.render(write_still=True)

result = bpy.data.images.load(a.out + ".exr")
pixels = np.empty(width * height * 4, dtype=np.float32)
result.pixels.foreach_get(pixels)
pixels.reshape(height, width, 4)[::-1, :, :3].astype(np.float32).tofile(a.out)
print("cycles render saved", a.out)
