import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import math


def calibrate_hover(model: mujoco.MjModel, init_data: mujoco.MjData, site_id: int,
                    ctrl_ids: np.ndarray, max_motor: float = 13.0,
                    dt_steps: int = 120, vel_tol: float = 0.002) -> float:
    """Find per-rotor command that results in near-zero vertical velocity (hover).

    Binary search on motor command c in [0, max_motor]. For each candidate we run
    dt_steps physics steps and measure average vertical velocity of the leader site.
    Returns the command value `c_hover`.
    """
    init_qpos = init_data.qpos.copy()
    init_qvel = init_data.qvel.copy()
    init_act = init_data.act.copy() if model.na > 0 else None

    tmp = mujoco.MjData(model)

    low = 0.0
    high = max_motor

    for _ in range(20):
        mid = 0.5 * (low + high)

        # reset to initial state each trial
        tmp.qpos[:] = init_qpos
        tmp.qvel[:] = init_qvel
        if init_act is not None:
            tmp.act[:] = init_act
        mujoco.mj_forward(model, tmp)

        pos0 = tmp.site_xpos[site_id].copy()
        tmp.ctrl[ctrl_ids] = mid
        for _step in range(dt_steps):
            mujoco.mj_step(model, tmp)

        pos1 = tmp.site_xpos[site_id].copy()
        vz = (pos1[2] - pos0[2]) / (dt_steps * model.opt.timestep)

        if abs(vz) <= vel_tol:
            return float(mid)
        if vz > 0:
            high = mid
        else:
            low = mid

    return float(0.5 * (low + high))


def _normalize(v: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


def _vee(M: np.ndarray) -> np.ndarray:
    # inverse of hat() for a skew-symmetric matrix
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def _desired_rotation_from_force(F_des_world: np.ndarray, psi_des: float) -> np.ndarray:
    # R_des columns are body axes in world frame: [b1 b2 b3]
    b3_des, nF = _normalize(F_des_world)
    if nF <= 1e-9:
        b3_des = np.array([0.0, 0.0, 1.0])

    b1_ref = np.array([math.cos(psi_des), math.sin(psi_des), 0.0], dtype=float)
    b2_des, n2 = _normalize(np.cross(b3_des, b1_ref))
    if n2 <= 1e-9:
        b1_ref = np.array([1.0, 0.0, 0.0], dtype=float)
        b2_des, _ = _normalize(np.cross(b3_des, b1_ref))

    b1_des = np.cross(b2_des, b3_des)
    return np.column_stack([b1_des, b2_des, b3_des])


def quad_backstepping_control(
        *,
        pos: np.ndarray,
        vel: np.ndarray,
        R: np.ndarray,
        omega_body: np.ndarray,
        pos_des: np.ndarray,
        vel_des: np.ndarray,
        acc_des: np.ndarray,
        psi_des: float,
        m: float,
        gravity: np.ndarray,
        I_diag: np.ndarray,
        kp_xy: float,
        kd_xy: float,
        kp_z: float,
        kd_z: float,
        k_R: float,
        k_omega: float,
        arm_x: float,
        arm_y: float,
        yaw_coeff: float,
        max_motor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Leader/follower controller: position -> desired force -> attitude backstepping -> allocation.

    Returns:
      - motor forces u (4,) in Newtons, directly compatible with actuator ctrl
      - commanded acceleration a_cmd (3,) in world frame
    """
    e_pos = pos - pos_des
    e_vel = vel - vel_des

    a_cmd = acc_des.astype(float).copy()
    a_cmd[0:2] -= kp_xy * e_pos[0:2] + kd_xy * e_vel[0:2]
    a_cmd[2] -= kp_z * e_pos[2] + kd_z * e_vel[2]

    # desired net force in world: F = m*(a - g)
    F_des_world = m * (a_cmd - gravity)

    # thrust command (project desired force along current body z)
    b3 = R[:, 2]
    F_total = float(np.dot(F_des_world, b3))
    F_total = float(np.clip(F_total, 0.0, 4.0 * max_motor))

    # desired attitude from desired force direction + yaw
    R_des = _desired_rotation_from_force(F_des_world, psi_des)

    # attitude error in body frame
    e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)

    # backstepping: virtual angular velocity command, then torque
    omega_c = -k_R * e_R
    e_omega = omega_body - omega_c
    M = -k_omega * e_omega + np.cross(omega_body, I_diag * omega_body)

    # allocate to 4 rotors: [F; Mx; My; Mz]
    A = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [-arm_y, arm_y, arm_y, -arm_y],
        [arm_x, arm_x, -arm_x, -arm_x],
        [-yaw_coeff, yaw_coeff, -yaw_coeff, yaw_coeff],
    ], dtype=float)
    b = np.array([F_total, M[0], M[1], M[2]], dtype=float)

    try:
        u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        u, *_ = np.linalg.lstsq(A, b, rcond=None)

    u = np.clip(u, 0.0, max_motor)
    return u, a_cmd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "model" / "skydio_x2"

    print("Montando o sistema multi-agente...")

    # Cena base do ambiente.
    cena = mujoco.MjSpec.from_file(str(model_dir / "scene.xml"))

    # Dois drones independentes com nomes unicos ao anexar.
    lider = mujoco.MjSpec.from_file(str(model_dir / "x2.xml"))
    seguidor = mujoco.MjSpec.from_file(str(model_dir / "x2.xml"))

    ancora_lider = cena.worldbody.add_site(pos=[0, 0, 0.1])
    cena.attach(lider, site=ancora_lider, prefix="leader-")

    ancora_seguidor = cena.worldbody.add_site(pos=[1.5, 0, 0.1])
    cena.attach(seguidor, site=ancora_seguidor, prefix="follower-")

    model = cena.compile()

    # Map actuators by name (robust to ordering)
    leader_act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"leader-thrust{i}")
        for i in range(1, 5)
    ], dtype=int)
    follower_act_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"follower-thrust{i}")
        for i in range(1, 5)
    ], dtype=int)

    leader_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leader-x2")
    follower_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "follower-x2")

    # Optional: disable collisions between drones, keep floor collisions.
    disable_inter_drone_collisions = True
    if disable_inter_drone_collisions:
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_gid >= 0:
            model.geom_contype[floor_gid] = 4
            model.geom_conaffinity[floor_gid] = 3

        leader_geoms = np.where(model.geom_bodyid == leader_body_id)[0]
        follower_geoms = np.where(model.geom_bodyid == follower_body_id)[0]
        for gid in leader_geoms:
            if model.geom_contype[gid] != 0:
                model.geom_contype[gid] = 1
                model.geom_conaffinity[gid] = 4
        for gid in follower_geoms:
            if model.geom_contype[gid] != 0:
                model.geom_contype[gid] = 2
                model.geom_conaffinity[gid] = 4

    data = mujoco.MjData(model)

    print(f"Modelo compilado com {model.nq} qpos, {model.nv} qvel e {model.nu} atuadores.")
    print("Iniciando a simulacao em 3 segundos...")

    gravity = model.opt.gravity.copy()
    m_leader = float(model.body_mass[leader_body_id])
    m_follower = float(model.body_mass[follower_body_id])
    I_leader = model.body_inertia[leader_body_id].astype(float)
    I_follower = model.body_inertia[follower_body_id].astype(float)
    max_motor = float(model.actuator_ctrlrange[leader_act_ids[0], 1])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(3)
        # Prepare site ids for state reading
        leader_imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "leader-imu")
        follower_imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "follower-imu")

        # Ensure derived quantities (site_xpos, site_xmat, sensors) are up-to-date.
        mujoco.mj_forward(model, data)

        dt = model.opt.timestep
        prev_pos = {
            "leader": data.site_xpos[leader_imu_id].copy(),
            "follower": data.site_xpos[follower_imu_id].copy(),
        }

        # Keep initial spacing as the desired formation offset (avoids initial jump)
        formation_offset = prev_pos["follower"] - prev_pos["leader"]

        # Sensor ids for angular velocity (gyro). MuJoCo returns body-frame omega.
        leader_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "leader-body_gyro")
        follower_gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "follower-body_gyro")
        if leader_gyro_id < 0 or follower_gyro_id < 0:
            raise RuntimeError(
                "Gyro sensor nao encontrado. Verifique os nomes 'leader-body_gyro' e 'follower-body_gyro'."
            )

        def read_sensor_vec(sensor_id: int) -> np.ndarray:
            adr = int(model.sensor_adr[sensor_id])
            dim = int(model.sensor_dim[sensor_id])
            return data.sensordata[adr:adr + dim].copy()

        # Control gains (tune as needed)
        kp_xy = 4.0
        kd_xy = 2.5
        kp_z = 6.0
        kd_z = 4.0

        k_R = 10.0
        k_omega = 1.0

        # geometry from x2.xml (exact)
        arm_y = 0.18
        arm_x = 0.14
        yaw_coeff = 0.0201

        def leader_desired(t: float):
            # Circular trajectory in XY, fixed altitude
            r = 0.5
            w = 0.5
            x = r * math.cos(w * t)
            y = r * math.sin(w * t)
            z = 0.5
            vx = -r * w * math.sin(w * t)
            vy = r * w * math.cos(w * t)
            vz = 0.0
            ax = -r * w * w * math.cos(w * t)
            ay = -r * w * w * math.sin(w * t)
            az = 0.0
            return (np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az]))

        t = 0.0
        while viewer.is_running():
            # read positions
            leader_pos = data.site_xpos[leader_imu_id].copy()
            follower_pos = data.site_xpos[follower_imu_id].copy()

            # velocities by finite difference
            leader_vel = (leader_pos - prev_pos["leader"]) / dt
            follower_vel = (follower_pos - prev_pos["follower"]) / dt

            psi_des = 0.0

            # Leader reference trajectory
            pos_des, vel_des, acc_des = leader_desired(t)

            # Leader state (attitude + omega)
            R_leader = data.site_xmat[leader_imu_id].reshape(3, 3)
            omega_leader = read_sensor_vec(leader_gyro_id)

            u_leader, a_cmd_leader = quad_backstepping_control(
                pos=leader_pos,
                vel=leader_vel,
                R=R_leader,
                omega_body=omega_leader,
                pos_des=pos_des,
                vel_des=vel_des,
                acc_des=acc_des,
                psi_des=psi_des,
                m=m_leader,
                gravity=gravity,
                I_diag=I_leader,
                kp_xy=kp_xy,
                kd_xy=kd_xy,
                kp_z=kp_z,
                kd_z=kd_z,
                k_R=k_R,
                k_omega=k_omega,
                arm_x=arm_x,
                arm_y=arm_y,
                yaw_coeff=yaw_coeff,
                max_motor=max_motor,
            )

            # Follower consensus: track leader state + fixed formation offset
            follower_pos_des = leader_pos + formation_offset
            follower_vel_des = leader_vel
            follower_acc_des = a_cmd_leader

            R_follower = data.site_xmat[follower_imu_id].reshape(3, 3)
            omega_follower = read_sensor_vec(follower_gyro_id)

            u_follower, _a_cmd_follower = quad_backstepping_control(
                pos=follower_pos,
                vel=follower_vel,
                R=R_follower,
                omega_body=omega_follower,
                pos_des=follower_pos_des,
                vel_des=follower_vel_des,
                acc_des=follower_acc_des,
                psi_des=psi_des,
                m=m_follower,
                gravity=gravity,
                I_diag=I_follower,
                kp_xy=kp_xy,
                kd_xy=kd_xy,
                kp_z=kp_z,
                kd_z=kd_z,
                k_R=k_R,
                k_omega=k_omega,
                arm_x=arm_x,
                arm_y=arm_y,
                yaw_coeff=yaw_coeff,
                max_motor=max_motor,
            )

            data.ctrl[leader_act_ids] = u_leader
            data.ctrl[follower_act_ids] = u_follower

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            prev_pos["leader"] = leader_pos
            prev_pos["follower"] = follower_pos
            t += dt


if __name__ == "__main__":
    main()