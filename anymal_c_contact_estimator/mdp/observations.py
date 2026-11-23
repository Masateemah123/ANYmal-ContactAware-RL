# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
Observation utilities for the proprioceptive contact estimator.

Estimator inputs per timestep:
    - joint positions      (12)
    - joint velocities     (12)
    - foot positions       (12)  -> 4 feet x xyz, base-relative
    - foot velocities      (12)  -> 4 feet x xyz, world frame
Total input dimension: 48
IMPORTANT:
This file ONLY:
- builds estimator input features
    - exposes estimator outputs to PPO

The recurrent estimator execution itself happens inside:
    contact_estimator_event.py

This helps keeps recurrent state management OUTSIDE the observation manager.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# HELPERS
#______________________________________________________________

def _get_robot(env: ManagerBasedRLEnv):
    return env.scene["robot"]

# BUILD ONE ESTIMATOR INPUT FRAME
#______________________________________________________________

def build_contact_estimator_frame(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Builds ONE proprioceptive timestep for the estimator.
    Returns: Tensor shape: (num_envs, 48)

    Features:
        joint_pos  : (N, 12)  joint positions
        joint_vel  : (N, 12)  joint velocities
        foot_pos_b : (N, 12)  foot positions relative to base  [4 feet x xyz]
        foot_vel_w : (N, 12)  foot linear velocities world frame [4 feet x xyz]

    Attribute reference (verified from ArticulationData @property list):
        robot.data.body_state_w      (N, num_bodies, 13)  pos+quat+linvel+angvel
        robot.data.root_link_pose_w  (N, 7)               root pos+quat world frame
        robot.data.joint_pos         (N, num_joints)
        robot.data.joint_vel         (N, num_joints)
    """

    robot = _get_robot(env)
    # joint_ids cached at startup as env.contact_joint_ids
    #____________________________________________________________________________
    joint_pos = robot.data.joint_pos[:, env.contact_joint_ids]   # (N, 12)
    joint_vel = robot.data.joint_vel[:, env.contact_joint_ids]   # (N, 12)

    body_state_w = robot.data.body_state_w                       # (N, num_bodies, 13)

    # FOOT POSITIONS (transform to base frame)
    # ___________________________________________________________________________

    # root/base pose in world frame
    root_pos_w  = robot.data.root_link_pose_w[:, 0:3]   # (N, 3)
    root_quat_w = robot.data.root_link_pose_w[:, 3:7]   # (N, 4)

    # foot poses in world frame
    foot_pos_w  = body_state_w[:, env.contact_feet_ids, 0:3]  # (N, 4, 3)
    foot_quat_w = body_state_w[:, env.contact_feet_ids, 3:7]  # (N, 4, 4)

    # reshape so subtract_frame_transforms can process batches
    #_____________________________________________________________________________

    num_envs  = env.num_envs
    num_feet  = len(env.contact_feet_ids)

    # doing the same thing for root pose for each foot
    root_pos_expand  = root_pos_w.unsqueeze(1).expand(-1, num_feet, -1)
    root_quat_expand = root_quat_w.unsqueeze(1).expand(-1, num_feet, -1)

    # flatten:
    # (N,4,3) -> (N*4,3)
    root_pos_flat   = root_pos_expand.reshape(-1, 3)
    root_quat_flat  = root_quat_expand.reshape(-1, 4)

    foot_pos_flat_w  = foot_pos_w.reshape(-1, 3)
    foot_quat_flat_w = foot_quat_w.reshape(-1, 4)

    # transform feet from world frame -> base frame
    #_____________________________________________________________________________

    foot_pos_b_flat, _ = math_utils.subtract_frame_transforms(root_pos_flat,root_quat_flat,foot_pos_flat_w,foot_quat_flat_w)

    # back to (N,4,3)
    foot_pos_b = foot_pos_b_flat.reshape(num_envs, num_feet, 3)

    # FOOT VELOCITIES — transform world -> base frame
    # Must match preprocessing pipeline used during training
    # _____________________________________________________________________________

    foot_vel_w = body_state_w[:, env.contact_feet_ids, 7:10]   # (N, 4, 3)
    # flatten
    foot_vel_flat_w = foot_vel_w.reshape(-1, 3)

    # rotate world velocity -> base frame (quat_apply_inverse)
    foot_vel_b_flat = math_utils.quat_apply_inverse(
        root_quat_flat,
        foot_vel_flat_w
    )

    # reshape back
    foot_vel_b = foot_vel_b_flat.reshape(num_envs, num_feet, 3)

    # FLATTEN FOOT TENSORS
    # (N, 4, 3) -> (N, 12)
    # ______________________________________________________________________________
    foot_pos_flat = foot_pos_b.reshape(env.num_envs, -1)          # (N, 12)
    foot_vel_flat = foot_vel_b.reshape(env.num_envs, -1)          # (N, 12)

    # _____________________________________________________________________________
    # ASSEMBLE FRAME
    # must match the order the network was trained on:
    # verify against deep-contact-estimator data loader if unsure
    # _____________________________________________________________________________

    frame = torch.cat(
        [
            joint_pos,      # 12
            joint_vel,      # 12
            foot_pos_flat,  # 12
            foot_vel_flat,  # 12
        ],
        dim=-1,
    )# (N, 48)
    # print(frame[0])
    # print(frame.mean())
    # print(frame.std())
    
    # SHAPE GUARD
    # if this fires, one of the slices above returned wrong dims
    # __________________________________________________________________________
    assert frame.shape == (env.num_envs, 48), (
        f"[build_contact_estimator_frame] expected (N, 48), got {frame.shape}\n"
        f"  joint_pos: {joint_pos.shape}\n"
        f"  joint_vel: {joint_vel.shape}\n"
        f"  foot_pos:  {foot_pos_flat.shape}\n"
        f"  foot_vel:  {foot_vel_flat.shape}"
    )
    # print runtime ordering 
    # print(env.contact_joint_ids)
    # print(env.contact_feet_ids)
    return frame.float()

# # PPO OBSERVATION TERM
# # ____________________________________________________________
def estimated_contacts(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Expose estimator outputs to PPO policy.
    Returns:
        Tensor: (num_envs, 4)
    Contact order: [LF, RF, LH, RH]
    """

    if not hasattr(env, "estimated_contacts"):
        return torch.zeros((env.num_envs, 4),device=env.device,dtype=torch.float32)

    return env.estimated_contacts.float()

