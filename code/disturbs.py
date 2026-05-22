from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindParams:
    """Parameters for a simple wind disturbance model.

    Units:
      - steady_force_world, gust_force_dir_world: Newtons (N)
      - turbulence_sigma: standard deviation of additive force noise (N)
      - gust_freq_hz: cycles per second
    """

    steady_force_world: np.ndarray
    gust_force_dir_world: np.ndarray
    gust_amp: float
    gust_freq_hz: float
    turbulence_sigma: float


def wind_params(
    *,
    steady_force_world: np.ndarray | list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    gust_force_dir_world: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0),
    gust_amp: float = 0.0,
    gust_freq_hz: float = 0.2,
    turbulence_sigma: float = 0.0,
) -> WindParams:
    steady = np.asarray(steady_force_world, dtype=float).reshape(3)
    gust_dir = np.asarray(gust_force_dir_world, dtype=float).reshape(3)
    norm = float(np.linalg.norm(gust_dir))
    if norm > 1e-12:
        gust_dir = gust_dir / norm
    else:
        gust_dir = np.array([1.0, 0.0, 0.0], dtype=float)

    return WindParams(
        steady_force_world=steady,
        gust_force_dir_world=gust_dir,
        gust_amp=float(gust_amp),
        gust_freq_hz=float(gust_freq_hz),
        turbulence_sigma=float(turbulence_sigma),
    )


def sample_wind_force_world(*, t: float, rng: np.random.Generator, params: WindParams) -> np.ndarray:
    """Samples a wind force in world frame (N).

    Model: steady + sinusoidal gust + white-noise turbulence.
    """
    gust = (params.gust_amp * math.sin(2.0 * math.pi * params.gust_freq_hz * t)) * params.gust_force_dir_world
    turb = rng.normal(loc=0.0, scale=params.turbulence_sigma, size=3)
    return params.steady_force_world + gust + turb


def add_position_noise(
    *,
    pos_world: np.ndarray,
    rng: np.random.Generator,
    sigma: float | np.ndarray,
    clip: float | None = None,
) -> np.ndarray:
    """Adds zero-mean Gaussian noise to a 3D position measurement.

    Args:
      pos_world: (3,) position in meters.
      sigma: scalar sigma (m) or (3,) per-axis sigma.
      clip: optional abs-clip value (m) applied to noise.

    Returns:
      Noisy position (3,).
    """
    pos = np.asarray(pos_world, dtype=float).reshape(3)
    sig = np.asarray(sigma, dtype=float)
    if sig.ndim == 0:
        noise = rng.normal(loc=0.0, scale=float(sig), size=3)
    else:
        sig = sig.reshape(3)
        noise = rng.normal(loc=0.0, scale=sig, size=3)

    if clip is not None:
        c = float(clip)
        noise = np.clip(noise, -c, c)

    return pos + noise


def clear_applied_wrench(*, xfrc_applied: np.ndarray, body_id: int) -> None:
    """Clears previously applied external force/torque for a body."""
    xfrc_applied[body_id, :] = 0.0


def apply_wind_wrench(
    *,
    xfrc_applied: np.ndarray,
    body_id: int,
    force_world: np.ndarray,
    torque_world: np.ndarray | None = None,
) -> None:
    """Applies a force/torque wrench to a body via MuJoCo's xfrc_applied.

    Args:
      xfrc_applied: data.xfrc_applied view.
      body_id: body index.
      force_world: (3,) force in world frame [N].
      torque_world: optional (3,) torque in world frame [N*m].

    Note:
      xfrc_applied is persistent for the duration of a step; you should clear
      it each loop if you want time-varying forces.
    """
    xfrc_applied[body_id, 0:3] = np.asarray(force_world, dtype=float).reshape(3)
    if torque_world is None:
        xfrc_applied[body_id, 3:6] = 0.0
    else:
        xfrc_applied[body_id, 3:6] = np.asarray(torque_world, dtype=float).reshape(3)
