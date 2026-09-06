"""Pure data-contract helpers shared by collection and export."""

from __future__ import annotations

from typing import Sequence

import numpy as np


STATE_NAMES = (
    "prev_surge",
    "prev_sway",
    "prev_heave",
    "prev_yaw",
    "dvl_vx",
    "dvl_vy",
    "dvl_vz",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "depth_m",
    "altitude_m",
    "ego_valid",
    "buoy_release_valid",
    "dvl_velocity_valid",
    "altitude_valid",
)

ACTION_NAMES = ("surge", "sway", "heave", "yaw")

# MAVROS OverrideRCIn uses zero-based array offsets. ArduSub channels are
# forward=5, lateral=6, throttle=3, yaw=4.
DEFAULT_ACTION_CHANNEL_INDICES = (4, 5, 2, 3)
RC_RELEASE = 0
RC_NO_CHANGE = 65535


def normalize_quaternion_wxyz(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float32)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected four quaternion elements, got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-6:
        raise ValueError("Quaternion is not finite and non-zero")
    return quaternion / norm


def update_normalized_rc_command(
    channels: Sequence[int],
    previous: Sequence[float],
    channel_indices: Sequence[int] = DEFAULT_ACTION_CHANNEL_INDICES,
    neutral_pwm: int = 1500,
    pwm_span: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert selected RC PWM channels to normalized body commands.

    CHAN_NOCHANGE and CHAN_RELEASE do not create fake action labels. Their
    values retain the previous command and are reported as invalid in the
    returned update mask.
    """
    if pwm_span <= 0:
        raise ValueError("pwm_span must be positive")
    command = np.asarray(previous, dtype=np.float32).copy()
    if command.shape != (4,):
        raise ValueError(f"Expected a four-axis previous command, got {command.shape}")
    updated = np.zeros(4, dtype=np.float32)

    for axis, channel_index in enumerate(channel_indices):
        if channel_index < 0 or channel_index >= len(channels):
            continue
        pwm = int(channels[channel_index])
        if pwm in (RC_RELEASE, RC_NO_CHANGE):
            continue
        command[axis] = np.clip((pwm - neutral_pwm) / float(pwm_span), -1.0, 1.0)
        updated[axis] = 1.0

    return command, updated


def build_state(
    previous_command: Sequence[float],
    dvl_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    linear_acceleration: Sequence[float],
    attitude_wxyz: Sequence[float],
    depth_m: float,
    altitude_m: float,
    validity: Sequence[float],
) -> np.ndarray:
    state = np.concatenate(
        [
            np.asarray(previous_command, dtype=np.float32),
            np.asarray(dvl_velocity, dtype=np.float32),
            np.asarray(angular_velocity, dtype=np.float32),
            np.asarray(linear_acceleration, dtype=np.float32),
            normalize_quaternion_wxyz(attitude_wxyz),
            np.asarray([depth_m, altitude_m], dtype=np.float32),
            np.asarray(validity, dtype=np.float32),
        ]
    )
    if state.shape != (len(STATE_NAMES),):
        raise ValueError(f"Expected {len(STATE_NAMES)} state values, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("State contains a non-finite value")
    return state
