"""Render the 50 s quadrotor flight of ``../rviz_drone/`` with Isaac Sim 6.0.1 (kinematic replay).

The flight is generated in memory by the simulator of ``../rviz_drone/`` (``flight.run`` with a
fixed seed) before Isaac Sim starts. Isaac Sim only renders: the timeline is never played, no
physics step runs, and the pose of the quadrotor prim is written from the simulator truth for each
frame (conversion in ``usd_pose.py``; cameras, propellers, trail and overlay in ``replay.py``).

Modes:
  (default)          PNG sequence of [--start, --end] at --fps with overlay, optionally an MP4 (--mp4)
  --stills T:CAM,..  single frames at the given times and cameras
  --verify-heading   pure +x flight and pure yaw seen from above; measures the rendered heading
  --inspect          hover, +x translation, +90 deg yaw and t = 36 s from the side; tilt from the
                     propeller prims (USD) and from the drone silhouette (image)
  --gui              window with real-time replay in a loop until the window is closed

Run through the isolation wrapper only:
    source simulation/isaac_render/env.sh
    OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON" simulation/isaac_render/render_flight.py [options]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("RVIZ_DRONE_REPO_ROOT", HERE.parents[1])).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "simulation" / "rviz_drone"))

import asset_alignment  # noqa: E402
import replay  # noqa: E402
from usd_pose import enu_flu_to_usd, quat_normalize, rotmat, usd_matrix  # noqa: E402
from rviz_drone import flight  # noqa: E402
from rviz_drone.quadrotor import QuadParams  # noqa: E402
from rviz_drone import reference as references  # noqa: E402

ASSETS = Path(os.environ.get("ISAACSIM601_ROOT", Path.home() / "isaacsim-6.0.1-ens491")) / "assets"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--gui", action="store_true", help="open a window and replay in real time")
parser.add_argument("--stills", default="", help="comma-separated TIME:CAMERA list, e.g. 18:chase,33:topdown")
parser.add_argument("--verify-heading", action="store_true")
parser.add_argument("--inspect", action="store_true",
                    help="side and oblique views of hover, +x translation, +90 deg yaw and t = 36 s")
parser.add_argument("--camera", choices=replay.CAMERAS, default="chase")
parser.add_argument("--env", choices=("field", "terrain", "grid"), default="field",
                    help="field: Isaac flat terrain with grass; terrain: Isaac rough terrain with grass; "
                         "grid: plain checkerboard (fallback)")
parser.add_argument("--sky", choices=("hdri", "plain"), default="hdri")
parser.add_argument("--sun", type=float, default=3000.0, help="distant light intensity")
parser.add_argument("--fill", type=float, default=800.0,
                    help="fill light intensity (no shadow, from behind the camera)")
parser.add_argument("--sky-intensity", type=float, default=400.0, help="dome light intensity")
parser.add_argument("--sun-elevation", type=float, default=47.9,
                    help="[deg] elevation of the sun in the HDRI (brightest pixel)")
parser.add_argument("--sun-azimuth", type=float, default=146.5,
                    help="[deg] from east (+x) towards north (+y); matched to the shadow of the sky light")
parser.add_argument("--start", type=float, default=0.0)
parser.add_argument("--end", type=float, default=50.0)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--mp4", action="store_true", help="also encode the PNG sequence with /usr/bin/ffmpeg")
parser.add_argument("--out", default=str(HERE / "frames"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trajectory", choices=sorted(references.TRAJECTORIES), default=references.DEFAULT_TRAJECTORY)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--supersample", type=int, default=None,
                    help="render at this multiple of the output size and downscale (default: 2 for "
                         "realtime, 1 for pathtraced)")
parser.add_argument("--no-overlay", action="store_true")
parser.add_argument("--renderer", choices=("realtime", "pathtraced"), default="pathtraced",
                    help="headless only: RTX real-time (FXAA, supersampled) or path tracing")
parser.add_argument("--spp", type=int, default=64, help="path tracing: samples per pixel and frame")
args, kit_args = parser.parse_known_args()
if args.supersample is None:
    args.supersample = 2 if args.renderer == "realtime" else 1
if args.verify_heading:
    # Low-noise scene for the measurement: flat grid ground and a plain sky.
    args.env, args.sky = "grid", "plain"
# Only the Kit arguments (isolation settings of the wrapper) reach SimulationApp.
sys.argv = [sys.argv[0]] + kit_args

DRONE_USD = ASSETS / "ARL-Robot-1" / "arl_robot_1.usd"
TERRAIN_USD = {"field": (ASSETS / "Terrains" / "flat_plane.usd", "/World/GroundPlane"),
               "terrain": (ASSETS / "Terrains" / "rough_plane.usd", "/World/mesh")}
GRASS_MDL = ASSETS / "Materials" / "Grass_Countryside.mdl"
SKY_HDR = ASSETS / "Skies" / "kloofendal_48d_partly_cloudy_4k.hdr"
for f in [DRONE_USD] + ([TERRAIN_USD[args.env][0], GRASS_MDL] if args.env in TERRAIN_USD else []) + \
         ([SKY_HDR] if args.sky == "hdri" else []):
    if not f.is_file():
        sys.exit(f"asset not found: {f}")

# ---- Flight, generated in memory before Isaac Sim starts ------------------------------------
t_wall = time.monotonic()
result = flight.run(seed=args.seed, trajectory=args.trajectory)
TRAJ = references.get(args.trajectory)
P_REF = result["p_ref"]
T, P, V, Q, ROTOR = result["t"], result["p"], result["v"], result["q"], result["rotor"]
PARAMS = QuadParams()
SPIN = replay.rotor_spin_direction(PARAMS.allocation())
PROP_ANGLE = replay.prop_angles(T, ROTOR, PARAMS.mass * 9.81 / 4.0, SPIN)
CHASE_EYE, CHASE_TARGET = replay.chase_path(T, P, replay.yaw_from_quat(Q))
FIXED_CAMERAS = replay.fixed_cameras(P, sun_azimuth_deg=args.sun_azimuth)
print(f"RENDER flight: trajectory={args.trajectory} seed={args.seed} samples={len(T)} duration={T[-1]:.3f} s "
      f"generated in {time.monotonic() - t_wall:.1f} s; model m={PARAMS.mass} arm={PARAMS.arm}",
      flush=True)

from isaacsim import SimulationApp  # noqa: E402  (must come before any omni import)

simulation_app = SimulationApp({"headless": not args.gui, "width": args.width, "height": args.height})

import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402

KIT_VERSION = omni.kit.app.get_app().get_kit_version()
print("RENDER kit:", KIT_VERSION, flush=True)
settings = carb.settings.get_settings()
settings.set("/rtx/post/motionblur/enabled", False)  # no motion blur
settings.set("/rtx/post/histogram/enabled", False)  # fixed exposure, comparable frames

# ---- Stage ----------------------------------------------------------------------------------
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_grid(path: str, half: float = 30.0, tile: float = 1.0) -> None:
    """Checkerboard mesh at z = 0 (two grey tones per face); fallback ground."""
    n = int(round(2 * half / tile))
    xs = np.linspace(-half, half, n + 1)
    pts = [Gf.Vec3f(float(xs[i]), float(xs[j]), 0.0) for i in range(n + 1) for j in range(n + 1)]
    idx, colors = [], []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            idx += [a, a + n + 1, a + n + 2, a + 1]
            c = 0.42 if (i + j) % 2 == 0 else 0.28
            colors.append(Gf.Vec3f(c, c, c))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateFaceVertexCountsAttr([4] * (n * n))
    mesh.CreateExtentAttr([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, 0.0)])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                            UsdGeom.Tokens.uniform).Set(colors)


def make_mdl_material(path: str, mdl: Path, name: str, inputs: dict) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl)), "mdl")
    shader.SetSourceAssetSubIdentifier(name, "mdl")
    for key, (type_name, value) in inputs.items():
        shader.CreateInput(key, type_name).Set(value)
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                   material.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    return material


asset_info = {"drone": {"path": str(DRONE_USD), "sha256": sha256(DRONE_USD)}}
if args.env in TERRAIN_USD:
    # Only the ground prim of the terrain asset is referenced (the asset also carries render
    # settings, a light and a physics material); its material is replaced by an NVIDIA grass
    # material, projected in world space because the meshes have no texture coordinates.
    terrain_usd, terrain_prim = TERRAIN_USD[args.env]
    env_prim = stage.DefinePrim("/World/Environment")
    env_prim.GetReferences().AddReference(str(terrain_usd), terrain_prim)
    grass = make_mdl_material("/World/Looks/Grass", GRASS_MDL, "Grass_Countryside", {
        "project_uvw": (Sdf.ValueTypeNames.Bool, True),
        "world_or_object": (Sdf.ValueTypeNames.Bool, True),
        "texture_scale": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(0.25, 0.25)),
    })
    for prim in Usd.PrimRange(env_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(grass, UsdShade.Tokens.strongerThanDescendants)
    asset_info["terrain"] = {"path": str(terrain_usd), "prim": terrain_prim, "sha256": sha256(terrain_usd)}
    asset_info["ground_material"] = {"path": str(GRASS_MDL), "sha256": sha256(GRASS_MDL)}
else:
    make_grid("/World/Ground")

if args.sky == "hdri":
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(SKY_HDR)))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    asset_info["sky"] = {"path": str(SKY_HDR), "sha256": sha256(SKY_HDR)}
else:
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
dome.CreateIntensityAttr(args.sky_intensity)
# The sky lights the scene but casts no shadow, so that the distant light gives the only (soft)
# shadow; with both, the sun in the image gave a second shadow in another direction.
UsdLux.ShadowAPI.Apply(dome.GetPrim()).CreateShadowEnableAttr(False)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(args.sun)
sun.CreateAngleAttr(2.5)  # [deg] apparent size; larger than the real sun for softer shadow edges
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
# The distant light points along its local -z; it is turned so that it comes from the sun of the
# sky image (one shadow direction for both lights).
_el, _az = np.radians(args.sun_elevation), np.radians(args.sun_azimuth)
SUN_DIR = np.array([np.cos(_el) * np.cos(_az), np.cos(_el) * np.sin(_az), np.sin(_el)])  # towards the sun
_m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.0), Gf.Vec3d(*map(float, -SUN_DIR)), Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
UsdGeom.Xformable(sun).AddTransformOp().Set(_m)

# Fill light: a distant light without shadow that shines along the view direction of the active
# camera (updated in set_frame), so that the side of the drone seen by the camera is never black
# when the drone is between the camera and the sun.
fill = UsdLux.DistantLight.Define(stage, "/World/Fill")
fill.CreateIntensityAttr(args.fill)
fill.CreateAngleAttr(10.0)
fill.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
UsdLux.ShadowAPI.Apply(fill.GetPrim()).CreateShadowEnableAttr(False)
fill_op = UsdGeom.Xformable(fill).AddTransformOp()

# Upcoming route: the reference route of the next ROUTE_S seconds as a faint thin line (the flown
# path is shown only by the trail behind the drone).
ROUTE_S, ROUTE_N = 8.0, 81
path_curve = UsdGeom.BasisCurves.Define(stage, "/World/UpcomingRoute")
path_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
route_pts_attr = path_curve.CreatePointsAttr([Gf.Vec3f(0.0)] * ROUTE_N)
path_curve.CreateCurveVertexCountsAttr([ROUTE_N])
ROUTE_WIDTH = 0.005  # [m] trail: 0.014 at the drone, tapering to 0.002
route_width_attr = path_curve.CreateWidthsAttr([ROUTE_WIDTH])
path_curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
path_curve.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.92, 1.0)])


def preview_material(path: str, diffuse, emissive=None, roughness: float = 0.6,
                     opacity: float = 1.0) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


# Route: thin, pale, partly transparent. Trail: wider, self-lit orange, so that it stands out from
# the route in light and in shadow; its width tapers with age.
UsdShade.MaterialBindingAPI.Apply(path_curve.GetPrim()).Bind(
    preview_material("/World/Looks/Route", (0.85, 0.9, 1.0), emissive=(0.25, 0.28, 0.35), roughness=0.8,
                     opacity=0.6))

TRAIL_N = 41
trail_curve = UsdGeom.BasisCurves.Define(stage, "/World/Trail")
trail_curve.CreateTypeAttr(UsdGeom.Tokens.linear)
trail_curve.CreateCurveVertexCountsAttr([TRAIL_N])
trail_curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
trail_pts_attr = trail_curve.CreatePointsAttr([Gf.Vec3f(0.0)] * TRAIL_N)
trail_widths_attr = trail_curve.CreateWidthsAttr([0.01] * TRAIL_N)
UsdShade.MaterialBindingAPI.Apply(trail_curve.GetPrim()).Bind(
    preview_material("/World/Looks/Trail", (1.0, 0.35, 0.0), emissive=(1.0, 0.3, 0.0), roughness=0.4))

# Quadrotor: the body prim carries the pose (usd_pose.py); the alignment prim below it carries the
# fixed asset-to-body rotation (asset_alignment.py); the asset is referenced below that unchanged.
asset_stage = Usd.Stage.Open(str(DRONE_USD))
print(f"RENDER drone asset: {DRONE_USD} metersPerUnit={UsdGeom.GetStageMetersPerUnit(asset_stage)} "
      f"upAxis={UsdGeom.GetStageUpAxis(asset_stage)} defaultPrim={asset_stage.GetDefaultPrim().GetPath()}",
      flush=True)
del asset_stage
body = UsdGeom.Xform.Define(stage, "/World/Quadrotor")
op_t = body.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
op_q = body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
alignment = UsdGeom.Xform.Define(stage, "/World/Quadrotor/alignment")
_qa = asset_alignment.ASSET_TO_BODY
alignment.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Quatd(float(_qa[0]), Gf.Vec3d(*map(float, _qa[1:]))))
ASSET_ROOT = "/World/Quadrotor/alignment/asset"
asset_prim = stage.DefinePrim(ASSET_ROOT)
asset_prim.GetReferences().AddReference(str(DRONE_USD))

# The asset is also a physics articulation (rigid bodies, joints, collisions). The replay is
# kinematic, so the physics description is removed in this stage (the asset file is unchanged):
# applied Physics/Physx schemas are deleted and joint and collision prims deactivated. (Removing
# physics alone did not cure the delayed visuals; turning instancing off above did.)
# The visuals of the asset are instanceable prims. With instancing, the rendered asset did not follow
# a change of the body pose until a later change inside the asset (inspection views showed the pose
# of the previous view while the nose marker was correct), so instancing is turned off in this stage.
deinstanced = 0
for prim in list(Usd.PrimRange(asset_prim)):
    if prim.IsValid() and prim.IsInstanceable():
        prim.SetInstanceable(False)
        deinstanced += 1
print(f"RENDER asset instancing turned off on {deinstanced} prims", flush=True)
removed_schemas, deactivated = 0, 0
for prim in list(Usd.PrimRange(asset_prim)):
    if not prim.IsValid():  # below a prim deactivated in an earlier iteration
        continue
    for schema in list(prim.GetAppliedSchemas()):
        if schema.startswith(("Physics", "Physx", "PhysX")):
            prim.RemoveAppliedSchema(schema)
            removed_schemas += 1
    if prim.GetTypeName().startswith("Physics") and prim.GetTypeName().endswith("Joint") or \
            prim.GetName() in ("collisions", "joints"):
        prim.SetActive(False)
        deactivated += 1
left = [str(pr.GetPath()) for pr in Usd.PrimRange(asset_prim)
        if any(sc.startswith(("Physics", "Physx", "PhysX")) for sc in pr.GetAppliedSchemas())]
print(f"RENDER asset physics removed: {removed_schemas} applied schemas, {deactivated} prims deactivated, "
      f"remaining physics prims: {len(left)}", flush=True)
if left:
    print("RENDER FAILED: physics schemas remain on " + ", ".join(left[:5]), flush=True)
    simulation_app.close()
    sys.exit(1)

# Check of the alignment (body prim at identity here): propeller centres and spin axes in the body
# frame must be those of the model.
xf_check = UsdGeom.XformCache()
align_err = 0.0
prop_ops, prop_base = [], []
for i, name in enumerate(replay.ROTOR_PRIMS):
    prim = stage.GetPrimAtPath(f"{ASSET_ROOT}/{name}")
    if not prim:
        sys.exit(f"propeller prim {name} not found in the asset")
    m_prop = np.array(xf_check.GetLocalToWorldTransform(prim))
    c, axis = m_prop[3, :3], m_prop[2, :3] / np.linalg.norm(m_prop[2, :3])
    expected = PARAMS.rotor_positions()[i]
    align_err = max(align_err, float(np.abs(c - expected).max()), float(np.abs(axis - [0, 0, 1]).max()))
    print(f"RENDER rotor {i} {name}: body frame ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) m, spin axis "
          f"({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}), model ({expected[0]:.4f}, {expected[1]:.4f}, "
          f"{expected[2]:.4f}) m, spin {int(SPIN[i]):+d}", flush=True)
    orient = UsdGeom.Xformable(prim).GetOrderedXformOps()
    op = next(o for o in orient if o.GetOpType() == UsdGeom.XformOp.TypeOrient)
    prop_ops.append(op)
    prop_base.append(op.Get())
print(f"RENDER alignment check: max deviation {align_err:.2e} (m or axis component)", flush=True)
if align_err > 1e-3:
    print("RENDER FAILED: asset alignment check", flush=True)
    simulation_app.close()
    sys.exit(1)
PROP_PRIMS = [stage.GetPrimAtPath(f"{ASSET_ROOT}/{name}") for name in replay.ROTOR_PRIMS]

# Axes for the inspection views (red x/east, green y/north, blue z/up, 0.4 m), hidden otherwise.
gizmo = UsdGeom.Xform.Define(stage, "/World/Axes")
gizmo_op = gizmo.AddTranslateOp()
for axis_name, color, offset in (("x", (0.9, 0.1, 0.1), (0.2, 0, 0)), ("y", (0.1, 0.8, 0.1), (0, 0.2, 0)),
                                 ("z", (0.1, 0.3, 0.95), (0, 0, 0.2))):
    cyl = UsdGeom.Cylinder.Define(stage, f"/World/Axes/{axis_name}")
    cyl.CreateAxisAttr(axis_name.upper())
    cyl.CreateRadiusAttr(0.012)
    cyl.CreateHeightAttr(0.4)
    cyl.AddTranslateOp().Set(Gf.Vec3d(*offset))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
gizmo.MakeInvisible()

# Front marker: red arrow along +x of the body (FLU forward).
NOSE_COLOR = Gf.Vec3f(0.9, 0.05, 0.03)
nose = UsdGeom.Xform.Define(stage, "/World/Quadrotor/nose")
shaft = UsdGeom.Cylinder.Define(stage, "/World/Quadrotor/nose/shaft")
shaft.CreateAxisAttr(UsdGeom.Tokens.x)
shaft.CreateRadiusAttr(0.012)
shaft.CreateHeightAttr(0.12)
shaft.AddTranslateOp().Set(Gf.Vec3d(0.31, 0.0, 0.0))
head = UsdGeom.Cone.Define(stage, "/World/Quadrotor/nose/head")
head.CreateAxisAttr(UsdGeom.Tokens.x)
head.CreateRadiusAttr(0.04)
head.CreateHeightAttr(0.08)
head.AddTranslateOp().Set(Gf.Vec3d(0.41, 0.0, 0.0))
for g in (shaft, head):
    g.CreateDisplayColorAttr([NOSE_COLOR])

# Cameras.
cameras = {}
for name in replay.CAMERAS:
    cam = UsdGeom.Camera.Define(stage, f"/World/Camera_{name}")
    cam.CreateFocalLengthAttr(18.0 if name == "chase" else 24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
    cameras[name] = (cam, cam.AddTransformOp())
verify_cam = UsdGeom.Camera.Define(stage, "/World/Camera_verify")
verify_cam.CreateFocalLengthAttr(24.0)
verify_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
verify_cam_op = verify_cam.AddTransformOp()


def look_at(eye, target, up) -> Gf.Matrix4d:
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(*map(float, up)))
    return view.GetInverse()


def set_body(p: np.ndarray, q: np.ndarray) -> None:
    t, qu = enu_flu_to_usd(p, q)
    op_t.Set(Gf.Vec3d(*map(float, t)))
    op_q.Set(Gf.Quatd(float(qu[0]), Gf.Vec3d(*map(float, qu[1:]))))


# Line widths are multiplied by this factor for the fixed cameras, which see the whole flight area
# from tens of metres (set by the caller for each image; in the GUI from --camera).
LINE_SCALE = {"chase": 1.0, "overview": 6.0, "topdown": 6.0}
line_scale = LINE_SCALE[args.camera]
fill_camera = args.camera  # camera whose view direction the fill light follows (set per still)


def set_frame(k: int) -> None:
    """Drone pose, propellers, trail and cameras for truth sample k."""
    set_body(P[k], Q[k])
    for i, op in enumerate(prop_ops):
        a = float(PROP_ANGLE[k, i])
        base = prop_base[i]
        quat_type, vec_type = (Gf.Quatf, Gf.Vec3f) if isinstance(base, Gf.Quatf) else (Gf.Quatd, Gf.Vec3d)
        op.Set(base * quat_type(np.cos(a / 2.0), vec_type(0.0, 0.0, np.sin(a / 2.0))))
    route_pts_attr.Set([Gf.Vec3f(*map(float, x))
                        for x in replay.upcoming_route(T, P_REF, k, horizon_s=ROUTE_S, n=ROUTE_N)])
    pts, age = replay.trail(T, P, k, n=TRAIL_N)
    trail_pts_attr.Set([Gf.Vec3f(*map(float, x)) for x in pts])
    trail_widths_attr.Set([float(line_scale * (0.012 * (1.0 - a) + 0.002)) for a in age])
    route_width_attr.Set([float(line_scale * ROUTE_WIDTH)])
    for name, (_, op) in cameras.items():
        pose = replay.camera_pose(name, k, CHASE_EYE, CHASE_TARGET, FIXED_CAMERAS)
        op.Set(look_at(pose.eye, pose.target, pose.up))
        if name == fill_camera:
            d = replay.fill_direction(pose.eye, pose.target)
            fill_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.0), Gf.Vec3d(*map(float, d)),
                                                Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse())


# ---- Run-time check of the conversion against the USD transform ------------------------------
POSE_TOLERANCE = 1e-7  # measured 1.392e-08 (Gf quaternion-to-matrix conversion in double)
xf_cache = UsdGeom.XformCache()
max_err = 0.0
for k in np.linspace(0, len(T) - 1, 200).astype(int):
    set_body(P[k], Q[k])
    xf_cache.Clear()
    m_usd = np.array(xf_cache.GetLocalToWorldTransform(body.GetPrim()))
    t_, q_ = enu_flu_to_usd(P[k], Q[k])
    max_err = max(max_err, float(np.abs(m_usd - usd_matrix(t_, q_)).max()))
print(f"RENDER pose check: max |M_usd - M_expected| over 200 samples = {max_err:.3e} "
      f"(tolerance {POSE_TOLERANCE:.0e})", flush=True)
if max_err > POSE_TOLERANCE:
    print("RENDER FAILED: pose conversion check", flush=True)
    simulation_app.close()
    sys.exit(1)


def overlay(image, k: int, camera: str):
    from PIL import ImageDraw, ImageFont
    lines = replay.overlay_lines(float(T[k]), TRAJ.segment_at(float(T[k])).name,
                                 float(np.linalg.norm(V[k])), camera)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=max(14, args.height // 34))
    pad, line_h = 12, int(font.size * 1.35)
    width = max(int(draw.textlength(s, font=font)) for s in lines) + 2 * pad
    draw.rectangle([10, 10, 10 + width, 10 + 2 * pad // 2 + line_h * len(lines) + pad // 2],
                   fill=(0, 0, 0, 140))
    for i, s in enumerate(lines):
        draw.text((10 + pad, 10 + pad // 2 + i * line_h), s, font=font, fill=(255, 255, 255, 255))
    return image


def write_info(out: Path, extra: dict) -> None:
    info = {"seed": args.seed, "kit": KIT_VERSION, "width": args.width, "height": args.height,
            "rt_subframes": args.rt_subframes, "supersample": args.supersample, "renderer": args.renderer,
            "spp": args.spp if args.renderer == "pathtraced" else None, "env": args.env, "sky": args.sky, "sun": args.sun,
            "sky_intensity": args.sky_intensity, "sun_elevation": args.sun_elevation,
            "sun_azimuth": args.sun_azimuth, "assets": asset_info,
            "pose_check_max_error": max_err, "pose_tolerance": POSE_TOLERANCE}
    info.update(extra)
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")


if not args.gui:
    import omni.replicator.core as rep
    from PIL import Image

    # FXAA: spatial anti-aliasing only. The temporal methods (TAA, DLSS) blend earlier frames into
    # the image, which showed as a ghost of the drone at its earlier position.
    if args.renderer == "pathtraced":
        # Path tracing accumulates --spp samples per pixel for each frame (and anti-aliases by
        # itself); the accumulation restarts when the stage changes, so no earlier frame remains.
        rep.settings.set_render_pathtraced(samples_per_pixel=args.spp)
    else:
        rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    products = {}

    def capture(camera_path: str, settle_steps: int = 1):
        # One render product per camera, created on first use and kept until the end: a render
        # product created after another one was destroyed returned images of an earlier state.
        if camera_path not in products:
            rp = rep.create.render_product(camera_path, (args.width * args.supersample,
                                                          args.height * args.supersample))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach([rp])
            products[camera_path] = (rp, annot)
            # Warm-up of a new render product: its first images can show an earlier stage state.
            for _ in range(5):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
        rp, annot = products[camera_path]
        # The first step brings the stage changes of this frame to the renderer; without it the
        # first image after a change can show the previous state (observed with a new render product).
        for _ in range(settle_steps):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)
        data = np.asarray(annot.get_data())[..., :3]
        retries = 0
        while data.max() == 0 and retries < 5:
            # The first path-traced images after start-up were black (shaders still compiling).
            retries += 1
            rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=True)
            data = np.asarray(annot.get_data())[..., :3]
        if retries:
            print(f"RENDER note: black image, captured again {retries} time(s)", flush=True)
        img = Image.fromarray(data)
        if args.supersample > 1:
            img = img.resize((args.width, args.height), Image.Resampling.LANCZOS)
        return img

    def prime(camera_path: str) -> None:
        """One discarded image per camera before the real ones: the first path-traced image of a
        new render product stayed black even after repeated steps; after one stage change it is
        correct. The state is set again by the caller afterwards."""
        set_frame(0)
        capture(camera_path)
        print(f"RENDER primed {camera_path}", flush=True)

    def stats(img) -> str:
        a = np.asarray(img).astype(float)
        return f"mean={a.mean():.1f} saturated={100.0 * (a.max(axis=2) >= 250).mean():.2f}%"

    t_render = time.monotonic()
    if args.inspect:
        from PIL import ImageDraw, ImageFont
        idir = out / "inspect"
        idir.mkdir(exist_ok=True)
        for old in idir.glob("*.png"):
            old.unlink()
        cam_path = str(verify_cam.GetPath())
        focal, h_ap = 24.0, verify_cam.GetHorizontalApertureAttr().Get()
        identity = np.array([1.0, 0.0, 0.0, 0.0])

        def yaw_q(deg: float) -> np.ndarray:
            return np.array([np.cos(np.radians(deg) / 2), 0.0, 0.0, np.sin(np.radians(deg) / 2)])

        k36 = replay.sample_index(T, 36.0)
        bz36 = rotmat(Q[k36])[:, 2]
        tilt_axis = np.cross([0.0, 0.0, 1.0], bz36)
        tilt_axis /= np.linalg.norm(tilt_axis)
        up_z, up_y = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        h = np.array([0.0, 0.0, 1.5])
        # name, position, attitude, eye, target, up, side view (rotor plane seen edge-on)
        views = [
            ("hover_side", h, identity, h + [0, -3.0, 0], h, up_z, True),
            ("hover_oblique", h, identity, h + [-2.2, -2.2, 1.1], h, up_z, False),
            ("xplus_m0.8", h + [-0.8, 0, 0], identity, h + [0, -3.5, 0], h, up_z, True),
            ("xplus_0.0", h, identity, h + [0, -3.5, 0], h, up_z, True),
            ("xplus_p0.8", h + [0.8, 0, 0], identity, h + [0, -3.5, 0], h, up_z, True),
            ("yaw90_oblique", h, yaw_q(90.0), h + [-2.2, -2.2, 1.1], h, up_z, False),
            ("yaw90_top", h, yaw_q(90.0), h + [0, 0, 4.0], h, up_y, False),
            ("t36_side", P[k36], Q[k36], P[k36] - 3.0 * tilt_axis, P[k36], up_z, True),
            ("t36_chase", P[k36], Q[k36], CHASE_EYE[k36], CHASE_TARGET[k36], up_z, False),
        ]
        prime(cam_path)
        gizmo.MakeVisible()
        rows = []
        for name, p, q, eye, target, up, side in views:
            on_flight = name.startswith("t36")
            if on_flight:
                set_frame(k36)
                path_curve.MakeVisible() if name == "t36_chase" else path_curve.MakeInvisible()
                trail_curve.MakeVisible() if name == "t36_chase" else trail_curve.MakeInvisible()
            else:
                set_body(p, q)
                path_curve.MakeInvisible()
                trail_curve.MakeInvisible()
            gizmo_op.Set(Gf.Vec3d(*map(float, np.asarray(p) + [-0.6, -0.6, -0.7])))
            verify_cam_op.Set(look_at(eye, target, up))
            img_on = np.asarray(capture(cam_path, settle_steps=2)).astype(int)
            body.MakeInvisible()
            img_off = np.asarray(capture(cam_path, settle_steps=2)).astype(int)
            body.MakeVisible()
            # Drone pixels near the projected body origin (the shadow on the ground is excluded).
            centre = replay.project(np.asarray(p)[None, :], eye, target, up, focal, h_ap, args.width, args.height)[0]
            depth = float(np.dot(np.asarray(p) - eye, (np.asarray(target) - eye) / np.linalg.norm(np.asarray(target) - eye)))
            radius = 0.45 * focal / h_ap * args.width / depth
            vv, uu = np.mgrid[0:args.height, 0:args.width]
            window = (uu - centre[0]) ** 2 + (vv - centre[1]) ** 2 < radius ** 2
            mask = window & (np.abs(img_on - img_off).sum(axis=2) > 40)
            image_angle = replay.principal_angle_deg(mask) if mask.sum() > 50 else float("nan")
            # USD: spin axes and centres of the propeller prims, nose direction.
            xf = UsdGeom.XformCache()
            mats = [np.array(xf.GetLocalToWorldTransform(pr)) for pr in PROP_PRIMS]
            axes = np.array([m[2, :3] / np.linalg.norm(m[2, :3]) for m in mats])
            m_nose = np.array(xf.GetLocalToWorldTransform(nose.GetPrim()))
            r_sim = rotmat(q if not on_flight else Q[k36])
            p_sim = np.asarray(p if not on_flight else P[k36])
            usd_tilt = float(np.degrees(np.arccos(np.clip(axes[:, 2].mean(), -1, 1))))
            sim_tilt = replay.tilt_deg(r_sim)
            usd_yaw = float(np.degrees(np.arctan2(m_nose[0, 1], m_nose[0, 0])))
            sim_yaw = float(np.degrees(np.arctan2(r_sim[1, 0], r_sim[0, 0])))
            centres_err = max(float(np.abs(m[3, :3] - (p_sim + r_sim @ b)).max())
                              for m, b in zip(mats, PARAMS.rotor_positions()))
            row = {"view": name, "sim_tilt_deg": sim_tilt, "usd_prop_axis_tilt_deg": usd_tilt,
                   "rotors_up": bool(axes[:, 2].min() > 0), "sim_yaw_deg": sim_yaw, "usd_nose_yaw_deg": usd_yaw,
                   "usd_rotor_centre_error_m": centres_err, "side_view": side,
                   "image_rotor_plane_angle_deg": image_angle if side else None,
                   "drone_pixels": int(mask.sum())}
            rows.append(row)
            # Expected geometry from the simulator pose, drawn over the image: rotor quadrilateral
            # (cyan) and body z axis (yellow).
            img = Image.fromarray(img_on.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            rot = np.array([p_sim + r_sim @ b for b in PARAMS.rotor_positions()[[0, 2, 1, 3]]])
            uv = replay.project(rot, eye, target, up, focal, h_ap, args.width, args.height)
            draw.line([tuple(x) for x in np.vstack([uv, uv[:1]])], fill=(0, 230, 255), width=2)
            zz = replay.project(np.array([p_sim, p_sim + 0.35 * r_sim[:, 2]]), eye, target, up, focal, h_ap,
                                args.width, args.height)
            draw.line([tuple(zz[0]), tuple(zz[1])], fill=(255, 220, 0), width=3)
            font = ImageFont.load_default(size=20)
            text = (f"{name}: sim tilt {sim_tilt:.1f} deg, USD rotor-axis tilt {usd_tilt:.1f} deg, "
                    f"rotors up {row['rotors_up']}, sim yaw {sim_yaw:.1f}, nose yaw {usd_yaw:.1f}")
            if side:
                text += f", image rotor-plane angle {image_angle:.1f} deg"
            draw.rectangle([0, args.height - 34, args.width, args.height], fill=(0, 0, 0))
            draw.text((10, args.height - 30), text, font=font, fill=(255, 255, 255))
            img.save(idir / f"{name}.png")
            print(f"RENDER inspect {name:14s} sim tilt {sim_tilt:5.1f} | USD rotor-axis tilt {usd_tilt:5.1f} "
                  f"rotors up {row['rotors_up']} | sim yaw {sim_yaw:6.1f} nose yaw {usd_yaw:6.1f} | rotor centre "
                  f"error {centres_err:.1e} m | image angle {image_angle if side else float('nan'):6.1f} "
                  f"(pixels {int(mask.sum())})", flush=True)
        gizmo.MakeInvisible()
        path_curve.MakeVisible()
        trail_curve.MakeVisible()
        (idir / "inspect.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    elif args.verify_heading:
        # Synthetic poses (not the simulator flight): +x translation at zero attitude, then a pure
        # yaw rotation; a top-down camera with north up (east to the right) looks at them.
        vdir = out / "verify_heading"
        vdir.mkdir(exist_ok=True)
        for old in vdir.glob("*.png"):
            old.unlink()
        z0, cam_h = 1.5, 6.0
        eye = np.array([0.0, 0.0, z0 + cam_h])
        verify_cam_op.Set(look_at(eye, np.array([0.0, 0.0, z0]), np.array([0.0, 1.0, 0.0])))
        path_curve.MakeInvisible()
        trail_curve.MakeInvisible()
        f_px = 24.0 / verify_cam.GetHorizontalApertureAttr().Get() * args.width
        cases = [("x", x, 0.0) for x in np.arange(-1.5, 1.51, 0.5)] + \
                [("yaw", 0.0, yaw) for yaw in np.arange(0.0, 360.0, 30.0)]
        rows = []
        prime(str(verify_cam.GetPath()))
        for kind, x, yaw in cases:
            p = np.array([x, 0.0, z0])
            q = quat_normalize(np.array([np.cos(np.radians(yaw) / 2), 0.0, 0.0, np.sin(np.radians(yaw) / 2)]))
            set_body(p, q)
            nose.MakeVisible()
            with_nose = np.asarray(capture(str(verify_cam.GetPath()))).astype(int)
            nose.MakeInvisible()
            without = np.asarray(capture(str(verify_cam.GetPath()))).astype(int)
            # Nose pixels: red in the image with the nose and clearly different without it (the
            # difference alone also picks up render noise all over the image).
            r, g, b = with_nose[..., 0], with_nose[..., 1], with_nose[..., 2]
            red = (r > 120) & (r - np.maximum(g, b) > 70)
            mask = red & (np.abs(with_nose - without).sum(axis=2) > 60)
            vs, us = np.nonzero(mask)
            # Projection of the body origin (pinhole, square pixels, camera straight above).
            u_c = args.width / 2 + (p[0] - eye[0]) / cam_h * f_px
            v_c = args.height / 2 - (p[1] - eye[1]) / cam_h * f_px
            if len(us) == 0:
                rows.append({"case": kind, "x": x, "yaw": yaw, "pixels": 0})
                continue
            u_a, v_a = us.mean(), vs.mean()
            heading = np.degrees(np.arctan2(-(v_a - v_c), u_a - u_c)) % 360.0
            err = (heading - yaw + 180.0) % 360.0 - 180.0
            rows.append({"case": kind, "x": float(x), "yaw": float(yaw), "pixels": int(mask.sum()),
                         "u_centre": float(u_c), "u_nose": float(u_a), "v_nose": float(v_a),
                         "heading_deg": float(heading), "error_deg": float(err)})
            Image.fromarray(with_nose.astype(np.uint8)).save(vdir / f"{kind}_x{x:+.1f}_yaw{yaw:03.0f}.png")
            print(f"RENDER verify {kind:3s} x={x:+.1f} yaw={yaw:5.1f}: nose pixels={mask.sum():5d} "
                  f"u_centre={u_c:7.1f} u_nose={u_a:7.1f} heading={heading:6.1f} error={err:+5.1f} deg",
                  flush=True)
        nose.MakeVisible()
        errs = [abs(r["error_deg"]) for r in rows if "error_deg" in r]
        u_nose_x = [r["u_nose"] for r in rows if r["case"] == "x" and "u_nose" in r]
        moves_right = bool(np.all(np.diff(u_nose_x) > 0))
        print(f"RENDER verify summary: max |heading error| = {max(errs):.2f} deg over {len(errs)} poses; "
              f"+x flight moves to the right (east) in the image: {moves_right}", flush=True)
        (vdir / "verify_heading.json").write_text(json.dumps(
            {"rows": rows, "max_abs_error_deg": max(errs), "x_flight_moves_east": moves_right}, indent=2) + "\n")
    elif args.stills:
        sdir = out / "stills"
        sdir.mkdir(exist_ok=True)
        for cam_name in dict.fromkeys(item.split(":")[1] for item in args.stills.split(",")):
            prime(str(cameras[cam_name][0].GetPath()))
        for item in args.stills.split(","):
            time_s, cam_name = item.split(":")
            k = replay.sample_index(T, float(time_s))
            line_scale, fill_camera = LINE_SCALE[cam_name], cam_name
            set_frame(k)
            img = capture(str(cameras[cam_name][0].GetPath()), settle_steps=4)
            print(f"RENDER still t={T[k]:.2f} camera={cam_name} {stats(img)}", flush=True)
            if not args.no_overlay:
                img = overlay(img, k, cam_name)
            img.save(sdir / f"still_t{T[k]:05.2f}_{cam_name}.png")
        write_info(sdir, {"stills": args.stills})
    else:
        seq = out / f"seq_{args.camera}_{args.start:05.2f}_{args.end:05.2f}"
        if seq.exists():
            shutil.rmtree(seq)
        seq.mkdir()
        times = np.arange(args.start, args.end + 1e-9, 1.0 / args.fps)
        prime(str(cameras[args.camera][0].GetPath()))
        for n, time_s in enumerate(times):
            k = replay.sample_index(T, float(time_s))
            set_frame(k)
            img = capture(str(cameras[args.camera][0].GetPath()))
            if n % 50 == 0:
                print(f"RENDER frame {n}/{len(times)} t={T[k]:.2f} {stats(img)}", flush=True)
            if not args.no_overlay:
                img = overlay(img, k, args.camera)
            img.save(seq / f"frame_{n:05d}.png")
        extra = {"camera": args.camera, "start": args.start, "end": args.end, "fps": args.fps,
                 "frames": int(len(times))}
        if args.mp4:
            mp4 = seq.parent / f"{seq.name}.mp4"
            cmd = ["/usr/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", f"{args.fps}",
                   "-i", str(seq / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", "18", str(mp4)]
            subprocess.run(cmd, check=True)
            extra["mp4"] = str(mp4)
            print(f"RENDER mp4: {mp4} ({mp4.stat().st_size} bytes)", flush=True)
        write_info(seq, extra)
    for rp, annot in products.values():
        annot.detach()
        rp.destroy()
    print(f"RENDER done in {time.monotonic() - t_render:.1f} s -> {out}", flush=True)
else:
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_window

    get_active_viewport().camera_path = str(cameras[args.camera][0].GetPath())
    labels = []
    try:
        frame = get_active_viewport_window().get_frame("flight_overlay")
        with frame:
            with ui.VStack():
                ui.Spacer(height=64)  # below the viewport toolbar
                with ui.HStack(height=0):
                    ui.Spacer(width=16)
                    with ui.ZStack(width=360, height=112):
                        ui.Rectangle(style={"background_color": 0x99000000, "border_radius": 4})
                        with ui.HStack():
                            ui.Spacer(width=10)
                            with ui.VStack(spacing=2):
                                ui.Spacer(height=6)
                                for _ in range(4):
                                    labels.append(ui.Label("", height=0,
                                                           style={"font_size": 20, "color": 0xFFFFFFFF}))
                ui.Spacer()
        overlay_where = "viewport"
    except Exception as exc:  # fall back to a small window
        print(f"RENDER gui: viewport overlay unavailable ({exc}); using a window", flush=True)
        window = ui.Window("Flight", width=320, height=140)
        with window.frame:
            with ui.VStack(spacing=2):
                for _ in range(4):
                    labels.append(ui.Label(""))
        overlay_where = "window"
    print(f"RENDER gui: camera={args.camera} overlay={overlay_where}; replaying in a loop; "
          f"close the window to exit", flush=True)
    # The replay time is the wall-clock time since the start (modulo the flight duration); the
    # frame rate and the difference between replay steps and wall-clock steps are logged.
    t0 = time.monotonic()
    last_wall, last_s, frames, dts, max_step_err, report_at = None, None, 0, [], 0.0, t0 + 10.0
    while simulation_app.is_running():
        wall = time.monotonic()
        s = (wall - t0) % T[-1]
        if last_wall is not None:
            dt_wall, dt_s = wall - last_wall, s - last_s
            if dt_s >= 0.0:  # not at the wrap to the start of the flight
                max_step_err = max(max_step_err, abs(dt_s - dt_wall))
            dts.append(dt_wall)
        last_wall, last_s = wall, s
        frames += 1
        if wall >= report_at and dts:
            d = np.array(dts)
            print(f"RENDER gui rate: {len(d) / d.sum():.1f} frames/s over {d.sum():.1f} s, frame time "
                  f"median {1e3 * np.median(d):.1f} ms, max {1e3 * d.max():.1f} ms; "
                  f"max |replay step - wall step| {1e3 * max_step_err:.3f} ms", flush=True)
            dts, report_at = [], wall + 10.0
        k = replay.sample_index(T, s)
        set_frame(k)
        for label, text in zip(labels, replay.overlay_lines(float(T[k]), TRAJ.segment_at(float(T[k])).name,
                                                           float(np.linalg.norm(V[k])), args.camera)):
            label.text = text
        simulation_app.update()

simulation_app.close()
