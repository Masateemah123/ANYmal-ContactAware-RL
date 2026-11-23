# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

def feet_stance_width(env, asset_cfg: SceneEntityCfg, min_distance: float) -> torch.Tensor:
    """Penalize a stance narrower than ``min_distance``.

    The lateral (base-frame y) separation of the front and of the rear foot pair is
    compared against ``min_distance``, and only stances narrower than that are penalized.
    The term therefore sets a lower bound on the stance width — it does not prescribe a
    gait, tie the two body sides together or reward any particular foot placement above
    the bound.

    ``asset_cfg.body_names`` must list the four feet as
    (front-left, front-right, rear-left, rear-right) with ``preserve_order=True``.
    """
    asset = env.scene[asset_cfg.name]
    # feet position relative to the base, expressed in the base frame
    offset_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, offset_w.shape[1], -1)
    pos_b = quat_apply_inverse(quat.reshape(-1, 4), offset_w.reshape(-1, 3)).reshape(offset_w.shape)

    front = torch.abs(pos_b[:, 0, 1] - pos_b[:, 1, 1])
    rear = torch.abs(pos_b[:, 2, 1] - pos_b[:, 3, 1])
    return (min_distance - front).clip(min=0.0) + (min_distance - rear).clip(min=0.0)

# ___________________________________________________________________________________________________________________
# Added base height and base too low reward functions for the velocity env. 
# These are used in the velocity_env_cfg.py file to reward the robot for maintaining a certain height and penalize it 
# for dropping below a certain threshold.
# ____________________________________________________________________________________________________________________

def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward base height close to target height using L2 kernel."""
    asset = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / 0.01)  # 0.01 = std^2

def base_too_low(env, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base height below threshold."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < threshold).float()

def contact_estimator_consistency(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward consistency between estimated and ground truth foot contacts.
    reward = 1.0 → estimator perfectly matches GT on all 4 feet
    reward = 0.0 → estimator completely wrong on all 4 feet
    It rewards the policy when the contact estimator's predictions match the physics simulator's ground truth contacts.
    That is all it does. It does not tell the policy where to step. It does not tell the policy where not to step. 
    It does not encode terrain geometry, step height, or foothold quality in any way.
    """
    if not hasattr(env, "estimated_contacts") or not hasattr(env, "sensor_foot_ids"):
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]

    # sensor_foot_ids = [4, 12, 8, 16] — sensor-local indices for LF RF LH RH
    net_forces = contact_sensor.data.net_forces_w_history[
        :, :, env.sensor_foot_ids, :
    ]
    gt_contacts = (net_forces.norm(dim=-1).max(dim=1)[0] > 1.0).float()

    est = env.estimated_contacts
    reward = 1.0 - torch.abs(est - gt_contacts).mean(dim=1)
    return reward

# contact estimator as an 
# def contact_estimator_consistency(env: ManagerBasedRLEnv) -> torch.Tensor:
#     if not hasattr(env, "estimated_contacts") or not hasattr(env, "sensor_foot_ids"):
#         return torch.zeros(env.num_envs, device=env.device)

#     contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]
#     net_forces = contact_sensor.data.net_forces_w_history[
#         :, :, env.sensor_foot_ids, :
#     ]
#     gt_contacts = (net_forces.norm(dim=-1).max(dim=1)[0] > 10.0).float()

#     est = env.estimated_contacts
#     # Returns negative: 0 when perfect, -1 when completely wrong
#     return -torch.abs(est - gt_contacts).mean(dim=1)

# def pmets_fqf_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """
#     Foothold Quality Function (FQF) — PMETS core reward.
    
#     FQF(p) = w1*Q_prop(p) + w2*Q_geom(p) + w3*Q_edge(p)
    
#     All three terms computed purely from proprioception.
#     No heightscan or point cloud required.
    
#     Q_prop: proprioceptive marginal error (EST vs GT contact deviation)
#     Q_geom: terrain geometry proxy (foot height variance)  
#     Q_edge: edge risk penalty (uncertain contact at unexpected height)
    
#     Returns negative reward — higher FQF = worse foothold = more penalty.
#     """
#     if not hasattr(env, "contact_probs") or not hasattr(env, "sensor_foot_ids"):
#         return torch.zeros(env.num_envs, device=env.device)

#     robot  = env.scene["robot"]
#     sensor = env.scene["contact_forces"]

#     # ── WEIGHTS (tunable, from proposal) ─────────────────────
#     W1 = 0.4   # Q_prop weight
#     W2 = 0.3   # Q_geom weight  
#     W3 = 0.3   # Q_edge weight

#     # ── SHARED QUANTITIES ─────────────────────────────────────
#     p         = env.contact_probs.clamp(1e-6, 1-1e-6)  # (N, 4)
#     foot_z    = robot.data.body_state_w[:, env.contact_feet_ids, 2]  # (N, 4)
#     base_z    = robot.data.root_pos_w[:, 2:3]           # (N, 1)

#     net_forces = sensor.data.net_forces_w_history[:, :, env.sensor_foot_ids, :]
#     f_contact  = (net_forces.norm(dim=-1).max(dim=1)[0] > 10.0).float()  # (N, 4)

#     # ── Q_prop: PROPRIOCEPTIVE MARGINAL ERROR ─────────────────
#     # |estimated contact probability - actual contact|
#     # High when estimator disagrees with reality
#     Q_prop = torch.abs(p - f_contact).mean(dim=1)  # (N,)

#     # ── Q_geom: TERRAIN GEOMETRY PROXY ───────────────────────
#     # Foot height variance across all 4 feet
#     # High variance = uneven terrain = difficult geometry
#     # Normalised by 0.15m (typical step height on rough terrain)
#     foot_z_std = foot_z.std(dim=1)                          # (N,)
#     Q_geom = torch.clamp(foot_z_std / 0.15, 0.0, 1.0)      # (N,)

#     # ── Q_edge: EDGE RISK PENALTY ─────────────────────────────
#     # Uncertain contact (high entropy) at unexpected height
#     # Signature of partial foothold or edge contact
#     D_NOMINAL = 0.52
#     EPSILON   = 0.06   # height tolerance
#     SIGMA     = 0.08   # tanh scale

#     z_expected = base_z - D_NOMINAL
#     delta      = torch.abs(foot_z - z_expected)             # (N, 4)

#     # binary entropy of contact probability
#     H = -p * torch.log2(p) - (1-p) * torch.log2(1-p)       # (N, 4)

#     # edge risk: uncertain + unexpected height + actually in contact
#     deviation = torch.clamp(delta - EPSILON, min=0.0)
#     Q_edge_per_foot = f_contact * H * torch.tanh(deviation / SIGMA)
#     Q_edge = Q_edge_per_foot.mean(dim=1)                    # (N,)

#     # ── FQF COMPOSITE SCORE ───────────────────────────────────
#     FQF = W1 * Q_prop + W2 * Q_geom + W3 * Q_edge          # (N,) in [0,1]

#     # Negative reward — higher FQF = worse foothold quality = more penalty
#     return -FQF

# def contact_confidence_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
#     #"Black-box neural networks yield uncalibrated uncertainty" — this is the key point she highlighted for you. 
#     # Their solution is classical (GMM + GLRT with explicit uncertainty scores). Your CNN-GRU is a black box that 
#     # gives you a contact prediction but no confidence score. 
    
#     #Gini is the probability of a random sample being classified correctly when selected randomly. 
#     #Gini denotes the purity, whereas Gini impurity tells us about the impurity of nodes.
#     # so we use i to;  group similar classes together.
#     # it should incentivize to produce certain, high confident outputs. 
#     # To use this as a confidence reward, we want the reward to increase as confidence/certainty increases. 
#     # Therefore, we invert the Gini Impurity.
    
#     #reward ↑ only when p̂ is decisive AND agrees with f
#     # → p̂=0.5 gets penalised by confidence (=0) AND mediocre consistency (=0.5)
#     # → p̂=0.9 with f=0 gets penalised by consistency (=0.1) despite high confidence
#     # → only clean decisive correct contacts get full reward from both terms
        
#     """
#     Reward estimator confidence — penalises (grey-area uncertain) contacts based on contact probabilities/ slip.
#     R = 1 when all feet are clearly in contact or clearly in swing.
#     R = 0 when all feet are maximally uncertain (p=0.5).
    
#     Math: R_i = 1 - 4*p*(1-p)  (inverted Gini impurity)
#     """
#     if not hasattr(env, "estimated_contacts"):
#         return torch.zeros(env.num_envs, device=env.device)

#     p = env.estimated_contacts  # (N, 4)
#     confidence = 1.0 - 4.0 * p * (1.0 - p)  # (N, 4)
#     return confidence.mean(dim=1)             # (N,)

# def marginal_support_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """
#     Penalises loss of diagonal support during locomotion.
#     Based on partial foothold concept — Wiedebach et al. (2016).
    
#     During trot, 2 feet should always be confidently in contact.
#     Penalty scales with how far below 2 the confident support drops.
    
#     Math: R = -max(0, (S* - S) / S*)
#           S = sum of feet with p > 0.7
#           S* = 2 (expected trot support)
#     """
#     if not hasattr(env, "contact_probs"):
#         return torch.zeros(env.num_envs, device=env.device)

#     CONF_THRESHOLD = 0.7   # foot is confidently in contact
#     S_STAR         = 2.0   # expected trot support count

#     p = env.contact_probs                              # (N, 4)
#     confident = (p > CONF_THRESHOLD).float()           # (N, 4) binary
#     S = confident.sum(dim=1)                           # (N,) total support

#     penalty = torch.clamp(S_STAR - S, min=0.0) / S_STAR  # (N,) in [0,1]
#     return -penalty

# def terrain_uncertainty_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """
#     Penalises uncertain contact estimator predictions during actual ground contact.
#     High uncertainty during contact = ambiguous terrain = partial foothold signature.
    
#     Math: U_i = f_i * H(p_i)
#           H(p) = -p*log2(p) - (1-p)*log2(1-p)  [binary entropy]
#           R = -(1/4) * sum_i U_i
    
#     R = 0   → estimator confident on all contacts
#     R = -1  → estimator maximally uncertain on all contacts
#     """
#     if not hasattr(env, "contact_probs") or not hasattr(env, "sensor_foot_ids"):
#         return torch.zeros(env.num_envs, device=env.device)

#     sensor = env.scene["contact_forces"]
#     net_forces = sensor.data.net_forces_w_history[:, :, env.sensor_foot_ids, :]
#     f_contact  = (net_forces.norm(dim=-1).max(dim=1)[0] > 10.0).float()  # (N, 4)

#     p = env.contact_probs.clamp(1e-6, 1.0 - 1e-6)  # (N, 4) — avoid log(0)

#     # binary entropy per foot
#     H = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)  # (N, 4)

#     # only penalise during actual ground contact
#     uncertainty = f_contact * H  # (N, 4)

#     return -uncertainty.mean(dim=1)  # (N,)

# WHERE AND HOW EXACTLY DO I PUT THESE 
# torques = robot.data.applied_torque

# Then:

# LF = torques[:,0:3].abs().mean()
# RF = torques[:,3:6].abs().mean()
# LH = torques[:,6:9].abs().mean()
# RH = torques[:,9:12].abs().mean()
