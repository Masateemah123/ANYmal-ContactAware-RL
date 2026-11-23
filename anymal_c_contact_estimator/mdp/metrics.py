# isaaclab_experiments/anymal_c_contact_estimator/mdp/metrics.py
# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
Locomotion metrics logged to TensorBoard during training.
Not used as rewards — observation only.
Call log_locomotion_metrics() from a custom callback or
add it to update_contact_estimator after inference.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ANYMAL_C_MASS_KG = 30.0
GRAVITY           = 9.81


def compute_cost_of_transport(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Cost of Transport (CoT) = P / (m * g * v)

    P  = sum of |torque * joint_velocity| across all joints  (N,)
    v  = forward base speed in world frame                   (N,)

    Returns CoT per environment. (N,)
    Clipped to [0, 10] to avoid division-by-zero at standstill.
    """
    robot = env.scene["robot"]

    # joint torques (N, 12) and velocities (N, 12)
    torques = robot.data.applied_torque        # (N, 12)
    joint_vel = robot.data.joint_vel           # (N, 12)

    # mechanical power — sum of |τ × q̇| per env
    power = torch.sum(torch.abs(torques * joint_vel), dim=1)  # (N,)

    # forward speed — norm of base linear velocity in xy plane
    base_vel = robot.data.root_lin_vel_w[:, :2]  # (N, 2)
    speed = torch.norm(base_vel, dim=1)           # (N,)

    # CoT — clip speed to avoid div/0 at standstill
    speed_safe = speed.clamp(min=0.05)
    cot = power / (ANYMAL_C_MASS_KG * GRAVITY * speed_safe)  # (N,)

    return cot.clamp(0.0, 10.0)


def compute_stride_frequency(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Proxy for stride frequency using contact sensor air time.
    Shorter air time per step = higher frequency gait.

    Returns mean last_air_time across 4 feet per env. (N,)
    Lower = faster stepping = higher frequency.
    """
    if not hasattr(env, "sensor_foot_ids"):
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene["contact_forces"]
    # last_air_time shape: (N, num_bodies)
    air_time = sensor.data.last_air_time[:, env.sensor_foot_ids]  # (N, 4)
    return air_time.mean(dim=1)  # (N,)


def compute_diagonal_contact_symmetry(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Trot symmetry score — measures how well the diagonal pairs
    LF+RH and RF+LH alternate in antiphase.

    In a perfect trot:
      LF and RH are in contact simultaneously
      RF and LH are in contact simultaneously
      The two pairs alternate

    Score = |mean(p_LF + p_RH) - mean(p_RF + p_LH)| / 2
    Range [0, 1] — higher means stronger diagonal pairing.
    0 = no trot structure
    1 = perfect alternating diagonal trot
    """
    if not hasattr(env, "estimated_contacts"):
        return torch.zeros(env.num_envs, device=env.device)

    p = env.estimated_contacts  # (N, 4) — LF RF LH RH

    p_LF = p[:, 0]
    p_RF = p[:, 1]
    p_LH = p[:, 2]
    p_RH = p[:, 3]

    diagonal_A = (p_LF + p_RH) / 2.0  # (N,) — trot pair 1
    diagonal_B = (p_RF + p_LH) / 2.0  # (N,) — trot pair 2

    # symmetry score — how different are the two pairs
    # in trot one should be high while the other is low
    symmetry = torch.abs(diagonal_A - diagonal_B)  # (N,)
    return symmetry


def log_locomotion_metrics(env: ManagerBasedRLEnv, writer, step: int):
    """
    Log all locomotion metrics to TensorBoard.
    Call this every N steps from update_contact_estimator.
    """
    if writer is None:
        return

    cot    = compute_cost_of_transport(env)
    stride = compute_stride_frequency(env)
    trot   = compute_diagonal_contact_symmetry(env)

    robot   = env.scene["robot"]
    speed   = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    writer.add_scalar("Locomotion/cost_of_transport_mean",   cot.mean().item(),    step)
    writer.add_scalar("Locomotion/cost_of_transport_median", cot.median().item(),  step)
    writer.add_scalar("Locomotion/stride_air_time_mean",     stride.mean().item(), step)
    writer.add_scalar("Locomotion/trot_symmetry_score",      trot.mean().item(),   step)
    writer.add_scalar("Locomotion/forward_speed_mean",       speed.mean().item(),  step)

    # CoT vs speed scatter proxy — log at env 0 only
    writer.add_scalar("Locomotion/env0_cot",   cot[0].item(),   step)
    writer.add_scalar("Locomotion/env0_speed", speed[0].item(), step)