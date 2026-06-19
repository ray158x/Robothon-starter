"""
Robothon 2026 - Aegis 四足机器人绕 FF91 超级跑车跑一圈仿真

功能：
1. 加载 Aegis 机器人 URDF 模型
2. 创建逼真的 FF91 超级跑车模型（含四个轮子和车门）
3. 实现 Trot 步态行走控制
4. 机器人绕汽车跑一圈
5. 实时窗口显示 + 视频录制
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install mujoco numpy imageio[ffmpeg]\n\n"
        f"Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent.parent
URDF_PATH = ROOT / "assets" / "Aegis" / "urdf" / "Aegis_mujoco.urdf"
OUTPUT_VIDEO = ROOT / "outputs" / "my_demo.mp4"

DURATION_S = 40.0
FPS = 30
WALK_SPEED = 0.5
TURN_SPEED = 0.8

GAIT_FREQUENCY = 2.0
STEP_HEIGHT = 0.08

LEGS = ("FL", "FR", "RR", "RL")
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}

# FF91 汽车参数
CAR_CENTER = np.array([0.0, 0.0, 0.0])
CAR_BODY_LENGTH = 5.2
CAR_BODY_WIDTH = 2.0
CAR_BODY_HEIGHT = 0.6
CAR_WHEEL_RADIUS = 0.38
CAR_WHEEL_WIDTH = 0.28
CAR_COLOR_BODY = [0.92, 0.08, 0.08, 1.0]
CAR_COLOR_ROOF = [0.08, 0.08, 0.08, 1.0]
CAR_COLOR_GLASS = [0.25, 0.45, 0.65, 0.75]
CAR_COLOR_WHEEL = [0.05, 0.05, 0.05, 1.0]
CAR_COLOR_LIGHT = [1.0, 1.0, 0.9, 1.0]
CAR_COLOR_DOOR = [0.85, 0.05, 0.05, 1.0]
CAR_COLOR_WHEEL_RIM = [0.8, 0.8, 0.8, 1.0]

CIRCLE_RADIUS = 4.5
CIRCLE_LAPS = 1.0
CIRCLE_DURATION = 30.0  # 绕圈时长（秒）
JUMP_DURATION = 10.0    # 跳跃上车时长（秒）


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id >= 0:
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_addr] = value


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(URDF_PATH.as_posix())

    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]

    base = spec.body("BASE_LINK")
    if base is None:
        raise ValueError("Missing BASE_LINK body in Aegis URDF")
    base.add_freejoint(name="floating_base_joint")

    world = spec.worldbody

    # 环境光增强
    world.add_light(
        name="light_main",
        pos=[0.0, 0.0, 10.0],
        dir=[0.0, 0.0, -1.0],
        diffuse=[1.0, 1.0, 1.0],
        specular=[0.5, 0.5, 0.5],
        ambient=[0.5, 0.5, 0.5],
        castshadow=True,
    )
    world.add_light(
        name="light_fill",
        pos=[-8.0, -8.0, 8.0],
        dir=[1.0, 1.0, -1.0],
        diffuse=[0.6, 0.6, 0.6],
        specular=[0.3, 0.3, 0.3],
        ambient=[0.3, 0.3, 0.3],
        castshadow=False,
    )
    world.add_light(
        name="light_key",
        pos=[8.0, 8.0, 8.0],
        dir=[-1.0, -1.0, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.4, 0.4, 0.4],
        ambient=[0.2, 0.2, 0.2],
        castshadow=True,
    )

    # 地面（更亮）
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.25, 0.25, 0.28, 1],
        friction=[0.8, 0.02, 0.005],
    )

    # ===================== FF91 超级跑车 =====================
    
    # 车身主体
    world.add_geom(
        name="car_body",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.5 + CAR_BODY_HEIGHT / 2],
        size=[CAR_BODY_LENGTH / 2, CAR_BODY_WIDTH / 2, CAR_BODY_HEIGHT / 2],
        rgba=CAR_COLOR_BODY,
        friction=[0.5, 0.02, 0.005],
    )

    # 车顶
    world.add_geom(
        name="car_roof",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.5 + CAR_BODY_HEIGHT + 0.25],
        size=[CAR_BODY_LENGTH / 2.5, CAR_BODY_WIDTH / 2.2, 0.2],
        rgba=CAR_COLOR_ROOF,
        friction=[0.5, 0.02, 0.005],
    )

    # 四个车门
    # 左前门
    world.add_geom(
        name="car_door_front_l",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.8, 0.92, 0.5 + CAR_BODY_HEIGHT / 2],
        size=[1.0, 0.03, CAR_BODY_HEIGHT / 2 - 0.05],
        rgba=CAR_COLOR_DOOR,
        friction=[0.5, 0.02, 0.005],
    )
    # 左后门
    world.add_geom(
        name="car_door_rear_l",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.8, 0.92, 0.5 + CAR_BODY_HEIGHT / 2],
        size=[1.0, 0.03, CAR_BODY_HEIGHT / 2 - 0.05],
        rgba=CAR_COLOR_DOOR,
        friction=[0.5, 0.02, 0.005],
    )
    # 右前门
    world.add_geom(
        name="car_door_front_r",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.8, -0.92, 0.5 + CAR_BODY_HEIGHT / 2],
        size=[1.0, 0.03, CAR_BODY_HEIGHT / 2 - 0.05],
        rgba=CAR_COLOR_DOOR,
        friction=[0.5, 0.02, 0.005],
    )
    # 右后门
    world.add_geom(
        name="car_door_rear_r",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.8, -0.92, 0.5 + CAR_BODY_HEIGHT / 2],
        size=[1.0, 0.03, CAR_BODY_HEIGHT / 2 - 0.05],
        rgba=CAR_COLOR_DOOR,
        friction=[0.5, 0.02, 0.005],
    )

    # 车窗（车门上的小窗户）
    world.add_geom(
        name="car_window_front_l",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.8, 0.95, 0.5 + CAR_BODY_HEIGHT + 0.15],
        size=[0.6, 0.02, 0.1],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )
    world.add_geom(
        name="car_window_rear_l",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.8, 0.95, 0.5 + CAR_BODY_HEIGHT + 0.15],
        size=[0.6, 0.02, 0.1],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )
    world.add_geom(
        name="car_window_front_r",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.8, -0.95, 0.5 + CAR_BODY_HEIGHT + 0.15],
        size=[0.6, 0.02, 0.1],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )
    world.add_geom(
        name="car_window_rear_r",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.8, -0.95, 0.5 + CAR_BODY_HEIGHT + 0.15],
        size=[0.6, 0.02, 0.1],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )

    # 前挡风玻璃
    world.add_geom(
        name="car_windshield",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.5, 0.0, 0.5 + CAR_BODY_HEIGHT + 0.35],
        size=[0.05, CAR_BODY_WIDTH / 2.3, 0.15],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )

    # 后挡风玻璃
    world.add_geom(
        name="car_rearwindow",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-1.5, 0.0, 0.5 + CAR_BODY_HEIGHT + 0.35],
        size=[0.05, CAR_BODY_WIDTH / 2.3, 0.15],
        rgba=CAR_COLOR_GLASS,
        friction=[0.5, 0.02, 0.005],
    )

    # 四个车轮（带轮毂）
    wheel_positions = [
        [1.7, 1.05, CAR_WHEEL_RADIUS],   # 前轮左
        [1.7, -1.05, CAR_WHEEL_RADIUS],  # 前轮右
        [-1.7, 1.05, CAR_WHEEL_RADIUS],  # 后轮左
        [-1.7, -1.05, CAR_WHEEL_RADIUS], # 后轮右
    ]
    
    for i, pos in enumerate(wheel_positions):
        # 轮胎（外圈）
        world.add_geom(
            name=f"car_wheel_tire_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos,
            size=[CAR_WHEEL_RADIUS, CAR_WHEEL_WIDTH / 2],
            rgba=CAR_COLOR_WHEEL,
            friction=[0.8, 0.02, 0.005],
            quat=[0.7071, 0.0, 0.7071, 0.0],
        )
        # 轮毂（内圈）
        world.add_geom(
            name=f"car_wheel_rim_{i}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=pos,
            size=[CAR_WHEEL_RADIUS * 0.65, CAR_WHEEL_WIDTH / 2 + 0.02],
            rgba=CAR_COLOR_WHEEL_RIM,
            friction=[0.5, 0.02, 0.005],
            quat=[0.7071, 0.0, 0.7071, 0.0],
        )

    # 前大灯
    world.add_geom(
        name="car_headlight_l",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[2.6, 0.55, 0.7],
        size=[0.14, 0.14, 0.14],
        rgba=CAR_COLOR_LIGHT,
        friction=[0.5, 0.02, 0.005],
    )
    world.add_geom(
        name="car_headlight_r",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[2.6, -0.55, 0.7],
        size=[0.14, 0.14, 0.14],
        rgba=CAR_COLOR_LIGHT,
        friction=[0.5, 0.02, 0.005],
    )

    # 尾灯
    world.add_geom(
        name="car_taillight_l",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[-2.6, 0.55, 0.7],
        size=[0.12, 0.12, 0.12],
        rgba=[1.0, 0.05, 0.05, 1.0],
        friction=[0.5, 0.02, 0.005],
    )
    world.add_geom(
        name="car_taillight_r",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[-2.6, -0.55, 0.7],
        size=[0.12, 0.12, 0.12],
        rgba=[1.0, 0.05, 0.05, 1.0],
        friction=[0.5, 0.02, 0.005],
    )

    # 车头装饰条
    world.add_geom(
        name="car_grille",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[2.6, 0.0, 0.6],
        size=[0.03, CAR_BODY_WIDTH / 3, 0.15],
        rgba=[0.95, 0.95, 0.95, 1.0],
        friction=[0.5, 0.02, 0.005],
    )

    return spec.compile()


def compute_leg_joints(model: mujoco.MjModel, data: mujoco.MjData, time_s: float,
                       forward_speed: float, turn_speed: float) -> None:
    settle = smoothstep(0.0, 0.5, time_s)

    gait = 2.0 * math.pi * GAIT_FREQUENCY * time_s

    for leg in LEGS:
        phase = LEG_PHASE[leg]
        swing = math.sin(gait + phase)
        
        lift = (1.0 - swing) / 2.0
        step_h = STEP_HEIGHT * lift * settle

        stride = 0.05 * swing * forward_speed / WALK_SPEED
        
        if leg in ("FR", "RR"):
            stride += 0.02 * turn_speed / TURN_SPEED
        else:
            stride -= 0.02 * turn_speed / TURN_SPEED

        set_joint(model, data, f"{leg}_HIP_YAW", 0.0)
        set_joint(model, data, f"{leg}_HIP_ROLL", 0.0)
        set_joint(model, data, f"{leg}_HIP_PITCH", stride)
        set_joint(model, data, f"{leg}_THIGH_PITCH", -0.6 + step_h * 8.0)
        set_joint(model, data, f"{leg}_CALF_PITCH", 1.2 - step_h * 8.0)


def compute_trajectory(time_s: float) -> tuple[np.ndarray, float]:
    """计算轨迹：绕圈 + 跳上车
    
    返回：
        robot_pos: [x, y, z] 位置
        robot_yaw: 偏航角
    """
    if time_s <= CIRCLE_DURATION:
        # 第一阶段：绕圈
        progress = time_s / CIRCLE_DURATION
        angle = progress * 2.0 * math.pi * CIRCLE_LAPS
        
        x = CAR_CENTER[0] + CIRCLE_RADIUS * math.cos(angle)
        y = CAR_CENTER[1] + CIRCLE_RADIUS * math.sin(angle)
        z = 0.33
        
        yaw = angle + math.pi / 2
        
        return np.array([x, y, z]), yaw
    else:
        # 第二阶段：跳上车
        jump_progress = (time_s - CIRCLE_DURATION) / JUMP_DURATION
        jump_progress = smoothstep(0.0, 1.0, jump_progress)
        
        # 结束位置：车顶中央后方
        end_x = CAR_CENTER[0] - 1.0  # 车顶后方
        end_y = 0.0
        end_z = 0.5 + CAR_BODY_HEIGHT + 0.25 + 0.15  # 车顶上方
        
        # 起始位置（绕圈结束位置：0度，即右侧）
        start_x = CAR_CENTER[0] + CIRCLE_RADIUS
        start_y = CAR_CENTER[1]
        start_z = 0.33
        
        # 插值到车上
        x = start_x + (end_x - start_x) * jump_progress
        y = start_y + (end_y - start_y) * jump_progress
        
        # Z轴：先向上跳再落到车上
        if jump_progress < 0.5:
            # 前半段：向上跳
            jump_height = 1.2 * smoothstep(0.0, 0.5, jump_progress)
            z = start_z + jump_height
        else:
            # 后半段：落到车上
            z = end_z + 1.2 * (1.0 - smoothstep(0.5, 1.0, jump_progress))
        
        # 朝向：先切线方向，最后朝向车头
        if jump_progress < 0.8:
            angle = 2.0 * math.pi * CIRCLE_LAPS  # 0度位置
            yaw = angle + math.pi / 2  # 朝向切线
        else:
            yaw = math.pi  # 朝向车头方向
        
        return np.array([x, y, z]), yaw


def run() -> None:
    print("=" * 60)
    print("Robothon 2026 - Aegis 绕 FF91 跑一圈后跳上车")
    print("=" * 60)

    print("\n[1/4] 加载 MuJoCo 模型...")
    model = build_model()
    print("  模型构建成功")

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    print("[2/4] 预热仿真...")
    for _ in range(250):
        compute_leg_joints(model, data, data.time, WALK_SPEED, 0.0)
        mujoco.mj_forward(model, data)

    print(f"[3/4] 开始仿真 ({DURATION_S}s)...")
    print(f"  视频将保存到: {OUTPUT_VIDEO}")
    
    total_frames = int(DURATION_S * FPS)

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    
    frames = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 5.5
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -30.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]

        for frame_idx in range(total_frames):
            time_s = frame_idx / FPS
            progress = frame_idx / total_frames

            speed_factor = smoothstep(0.05, 0.15, progress) * smoothstep(0.9, 0.8, progress)
            
            robot_pos, robot_yaw = compute_trajectory(time_s)
            
            forward_speed = WALK_SPEED * speed_factor
            
            # 计算转向速度
            if time_s <= CIRCLE_DURATION:
                turn_speed = (2.0 * math.pi * CIRCLE_LAPS / CIRCLE_DURATION) * speed_factor
            else:
                turn_speed = 0.2 * speed_factor  # 跳上车时慢速转向

            data.qpos[:] = 0.0
            data.qpos[0:3] = robot_pos
            data.qpos[3:7] = [math.cos(robot_yaw / 2.0), 0.0, 0.0, math.sin(robot_yaw / 2.0)]

            compute_leg_joints(model, data, time_s, forward_speed, turn_speed)

            mujoco.mj_forward(model, data)

            viewer.cam.lookat[:] = [robot_pos[0], robot_pos[1], robot_pos[2]]
            viewer.sync()

            renderer.update_scene(data)
            frame = renderer.render()
            frames.append(frame)

            if frame_idx % (FPS * 2) == 0:
                if time_s <= CIRCLE_DURATION:
                    angle_deg = (time_s / CIRCLE_DURATION) * 360
                    print(f"  [绕圈] {time_s:.1f}s/{CIRCLE_DURATION}s | "
                          f"角度: {angle_deg:.0f}度 | 位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
                else:
                    jump_s = time_s - CIRCLE_DURATION
                    print(f"  [跳上车] {jump_s:.1f}s/{JUMP_DURATION}s | "
                          f"高度: {robot_pos[2]:.2f}m | 位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")

    print("\n[4/4] 保存视频...")
    try:
        iio.imwrite(OUTPUT_VIDEO, np.asarray(frames), fps=FPS, codec="libx264")
        print(f"[OK] Video saved: {OUTPUT_VIDEO}")
    except Exception as exc:
        fallback = OUTPUT_VIDEO.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=FPS)
        print(f"[WARN] Saved as GIF: {fallback}")
        print(f"   Reason: {exc}")

    print("\n仿真完成！")
    print("=" * 60)


if __name__ == "__main__":
    run()
