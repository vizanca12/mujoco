import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import math

import disturbs
from sim_logger import SimDataRecorder


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


def chain_adjacency(n: int) -> np.ndarray:
    """Directed chain graph adjacency matrix.

    Convention: A[i, j] > 0 means agent i receives info from agent j.

    For a chain (Drone1 -> Drone2 -> ... -> DroneN):
      A[i, i-1] = 1 for i >= 1.
    """
    if n < 2:

        raise ValueError("n must be >= 2")
    A = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        A[i, i - 1] = 1.0
    return A


def star_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        A[i, 0] = 1.0  # Todos os seguidores olham APENAS para o Líder
    return A


def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian L = D - A (row-sum degree for directed graphs)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    D = np.diag(A.sum(axis=1))
    return D - A


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    model_dir = base_dir / "model" / "skydio_x2"

    print("Montando o sistema multi-agente...")

    # -----------------
    # Multi-drone setup
    # -----------------
    # Use 4 ou 5 drones aqui.
    n_drones = 5
    if n_drones < 2:
        raise ValueError("n_drones must be >= 2")

    prefixes = [f"drone{i + 1}-" for i in range(n_drones)]  # Drone1-, Drone2-, ...

    spawn_spacing_x = 1.5
    spawn_z = 0.1

    # Cena base do ambiente.
    cena = mujoco.MjSpec.from_file(str(model_dir / "scene.xml"))

    # Instancia N drones independentes com nomes unicos via prefix.
    for i, prefix in enumerate(prefixes):
        drone_spec = mujoco.MjSpec.from_file(str(model_dir / "x2.xml"))
        anchor = cena.worldbody.add_site(pos=[spawn_spacing_x * i, 0.0, spawn_z])
        cena.attach(drone_spec, site=anchor, prefix=prefix)

    model = cena.compile()

    def checked_name2id(obj: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx < 0:
            raise RuntimeError(f"Nome nao encontrado no modelo: '{name}'")
        return int(idx)

    # IDs por drone (atuadores, corpo raiz, sites/sensores)
    act_ids: list[np.ndarray] = []
    body_ids = np.zeros(n_drones, dtype=int)
    imu_site_ids = np.zeros(n_drones, dtype=int)
    gyro_sensor_ids = np.zeros(n_drones, dtype=int)

    for i, prefix in enumerate(prefixes):
        act = np.array([
            checked_name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}thrust{k}")
            for k in range(1, 5)
        ], dtype=int)
        act_ids.append(act)

        body_ids[i] = checked_name2id(mujoco.mjtObj.mjOBJ_BODY, f"{prefix}x2")
        imu_site_ids[i] = checked_name2id(mujoco.mjtObj.mjOBJ_SITE, f"{prefix}imu")
        gyro_sensor_ids[i] = checked_name2id(mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}body_gyro")

    # -----------------
    # Disturbances setup
    # -----------------
    rng = np.random.default_rng(0)

    leader_idx = 0  # Drone1 = lider
    enable_leader_pos_noise = True
    leader_pos_noise_sigma_m = 0.01  # meters (std)
    leader_pos_noise_clip_m = 0.03
    use_noisy_leader_for_consensus = True

    enable_wind = True
    wind_target_indices = [1]  # ex: [1] -> aplica vento no Drone2
    wind = disturbs.wind_params(
        steady_force_world=(0.6, 0.0, 0.0),
        gust_force_dir_world=(1.0, 0.2, 0.0),
        gust_amp= 0.6,
        gust_freq_hz=0.25,
        turbulence_sigma=0.15,
    )

    # Optional: disable collisions between drones, keep floor collisions.
    disable_inter_drone_collisions = False
    if disable_inter_drone_collisions:
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_gid >= 0:
            # Allocate a dedicated collision bit for the floor, and one bit per drone.
            floor_contype = 1 << n_drones
            drone_bits = [1 << i for i in range(n_drones)]
            floor_conaffinity = (1 << n_drones) - 1

            model.geom_contype[floor_gid] = floor_contype
            model.geom_conaffinity[floor_gid] = floor_conaffinity

            prefix_to_bit = {prefix: bit for prefix, bit in zip(prefixes, drone_bits)}
            for gid in range(model.ngeom):
                if model.geom_contype[gid] == 0:
                    continue
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if not gname:
                    continue
                matched_bit = None
                for prefix, bit in prefix_to_bit.items():
                    if gname.startswith(prefix):
                        matched_bit = bit
                        break
                if matched_bit is None:
                    continue
                model.geom_contype[gid] = matched_bit
                model.geom_conaffinity[gid] = floor_contype

    data = mujoco.MjData(model)

    print(f"Modelo compilado com {model.nq} qpos, {model.nv} qvel e {model.nu} atuadores.")
    print("Iniciando a simulacao em 3 segundos...")

    gravity = model.opt.gravity.copy()

    m = model.body_mass[body_ids].astype(float)
    I_diag = model.body_inertia[body_ids].astype(float)
    max_motor = float(model.actuator_ctrlrange[act_ids[0][0], 1])

    dofadr = model.body_dofadr[body_ids].astype(int)

    # -----------------
    # Communication graph
    # -----------------
    # Drone3 segue Drone2 porque A[2,1] = 1 no grafo em cadeia.
    #A = chain_adjacency(n_drones)
    A = star_adjacency(n_drones)
    L = laplacian_from_adjacency(A)
    print("Adjacency A (i recebe de j):")
    print(A)
    print("Laplaciana L = D - A:")
    print(L)

    labels = [p[:-1] if p.endswith("-") else p for p in prefixes]
    recorder = SimDataRecorder(n_drones=n_drones, labels=labels)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(3)
        # Ensure derived quantities (site_xpos, site_xmat, sensors) are up-to-date.
        mujoco.mj_forward(model, data)

        dt = float(model.opt.timestep)

        pos0 = data.site_xpos[imu_site_ids].copy()

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
        a_cmd_prev = np.zeros((n_drones, 3), dtype=float)
        while viewer.is_running():
            # Clear previous applied wrenches, then apply wind to selected drones.
            for bid in body_ids:
                disturbs.clear_applied_wrench(xfrc_applied=data.xfrc_applied, body_id=int(bid))
            if enable_wind:
                for wi in wind_target_indices:
                    if wi < 0 or wi >= n_drones:
                        continue
                    F_wind = disturbs.sample_wind_force_world(t=t, rng=rng, params=wind)
                    disturbs.apply_wind_wrench(
                        xfrc_applied=data.xfrc_applied,
                        body_id=int(body_ids[wi]),
                        force_world=F_wind,
                    )

            # True state
            pos_true = data.site_xpos[imu_site_ids].copy()

            # Measured/communicated positions (inject noise on leader, optionally used by the graph)
            pos_meas = pos_true.copy()
            if enable_leader_pos_noise:
                pos_meas[leader_idx] = disturbs.add_position_noise(
                    pos_world=pos_true[leader_idx],
                    rng=rng,
                    sigma=leader_pos_noise_sigma_m,
                    clip=leader_pos_noise_clip_m,
                )

            pos_comm = pos_true.copy()
            if enable_leader_pos_noise and use_noisy_leader_for_consensus:
                pos_comm[leader_idx] = pos_meas[leader_idx]

            # Use MuJoCo-provided linear velocity from the freejoint (world frame).
            vel = np.zeros((n_drones, 3), dtype=float)
            for i in range(n_drones):
                adr = int(dofadr[i])
                vel[i] = data.qvel[adr:adr + 3]

            psi_des = 0.0

            u_cmd = [None] * n_drones
            a_cmd = np.zeros((n_drones, 3), dtype=float)
            pos_ref_all = np.zeros((n_drones, 3), dtype=float)

            # Leader reference trajectory
            pos_des_0, vel_des_0, acc_des_0 = leader_desired(t)

            for i in range(n_drones):
                R_i = data.site_xmat[imu_site_ids[i]].reshape(3, 3)
                omega_i = read_sensor_vec(int(gyro_sensor_ids[i]))

                if i == leader_idx:
                    pos_des_i = pos_des_0
                    vel_des_i = vel_des_0
                    acc_des_i = acc_des_0
                    pos_i = pos_meas[i]
                else:
                    nbrs = np.where(A[i] > 0.0)[0]
                    if nbrs.size == 0:
                        raise RuntimeError(f"Drone{i + 1} nao tem vizinhos de entrada no grafo (linha {i}).")
                    w = A[i, nbrs].astype(float)
                    w = w / float(w.sum())

                    # Formation offset along each communication edge: pos0[i] - pos0[j].
                    delta = pos0[i] - pos0[nbrs]
                    pos_ref_neighbors = pos_comm[nbrs] + delta

                    pos_des_i = (w[:, None] * pos_ref_neighbors).sum(axis=0)
                    vel_des_i = (w[:, None] * vel[nbrs]).sum(axis=0)
                    # acc_des_i = (w[:, None] * a_cmd_prev[nbrs]).sum(axis=0)
                    pos_i = pos_true[i]

                pos_ref_all[i] = pos_des_i

                u_i, a_i = quad_backstepping_control(
                    pos=pos_i,
                    vel=vel[i],
                    R=R_i,
                    omega_body=omega_i,
                    pos_des=pos_des_i,
                    vel_des=vel_des_i,
                    acc_des=acc_des_i,
                    psi_des=psi_des,
                    m=float(m[i]),
                    gravity=gravity,
                    I_diag=I_diag[i],
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
                u_cmd[i] = u_i
                a_cmd[i] = a_i

                data.ctrl[act_ids[i]] = u_i

            a_cmd_prev = a_cmd

            # Log before stepping: compare real position vs reference used in this iteration.
            recorder.log(t=t, pos_true=pos_true, pos_ref=pos_ref_all)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            t += dt

    # When the MuJoCo window is closed, generate plots.
    recorder.plot(title="Posicao: real (solido) vs referencia (tracejado)")
    # Cascata: como o Drone2 (idx=1) afeta o Drone3 (idx=2) via grafo.
    if n_drones >= 3:
        recorder.plot_cascade(pos0=pos0, A=A, leader_idx=leader_idx, src_idx=1, dst_idx=2)


if __name__ == "__main__":
    main()