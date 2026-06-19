from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install mujoco numpy imageio[ffmpeg]\n\n"
        f"Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent.parent
URDF_PATH = ROOT / "assets" / "Aegis" / "urdf" / "Aegis_mujoco.urdf"
OUTPUT_VIDEO = ROOT / "outputs" / "my_first_demo.mp4"


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.option.timestep = 0.002

    base = spec.body("BASE_LINK")
    if base is not None:
        base.add_freejoint(name="floating_base_joint")

    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.06, 0.07, 0.08, 1],
    )
    world.add_light(pos=[0, -1.0, 2.0], dir=[0, 0.3, -1], diffuse=[1, 1, 1])

    return spec.compile()


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[2] = 0.35

    gait = 2.0 * math.pi * 1.2 * time_s

    leg_joints = [
        "FL_HIP_JOINT", "FR_HIP_JOINT",
        "RL_HIP_JOINT", "RR_HIP_JOINT",
        "FL_KNEE_JOINT", "FR_KNEE_JOINT",
        "RL_KNEE_JOINT", "RR_KNEE_JOINT",
    ]

    phases = [0, math.pi, 0, math.pi, 0, math.pi, 0, math.pi]
    for joint_name, phase in zip(leg_joints, phases):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        if "HIP" in joint_name:
            data.qpos[qpos_addr] = 0.5 + 0.3 * math.sin(gait + phase)
        elif "KNEE" in joint_name:
            data.qpos[qpos_addr] = -1.0 + 0.3 * max(0.0, math.sin(gait + phase))

    mujoco.mj_forward(model, data)


def run():
    model = build_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    duration_s = 5.0
    fps = 30
    total_frames = int(duration_s * fps)

    frames = []
    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        apply_pose(model, data, time_s)
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(OUTPUT_VIDEO, np.asarray(frames), fps=fps, codec="libx264")
        print(f"✅ Demo video saved to: {OUTPUT_VIDEO}")
    except Exception as exc:
        fallback = OUTPUT_VIDEO.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        print(f"⚠️  Video fallback to GIF: {fallback}")
        print(f"   Reason: {exc}")


if __name__ == "__main__":
    run()