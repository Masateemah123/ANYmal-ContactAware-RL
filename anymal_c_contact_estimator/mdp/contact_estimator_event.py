import os
# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
This module handles: estimator initialization, temporal history buffering,recurrent inference execution"""

from __future__ import annotations
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_experiments.anymal_c_contact_estimator.contact_estimator_wrapper import (ContactEstimatorWrapper)
from isaaclab_experiments.anymal_c_contact_estimator.mdp.observations import (build_contact_estimator_frame)

def init_contact_estimator(env, env_ids=None):
    """Initialize contact estimator runtime. Called once at startup."""
    model_path = (os.getenv("CONTACTNET_CKPT", "../ContactNet/checkpoints/contactnet_48d_epoch19.ckpt"))
    #model_path = os.getenv("CONTACTNET_CKPT", "../ContactNet/checkpoints/contactnet_48d_epoch19.ckpt")
    env.contact_estimator = ContactEstimatorWrapper(model_path=model_path, num_envs=env.num_envs, device=env.device)
    #feet_cfg = SceneEntityCfg("robot",body_names=".*FOOT")
    feet_cfg = SceneEntityCfg("robot",body_names=".*foot")
    feet_cfg.resolve(env.scene)
    print("FOOT IDS:", feet_cfg.body_ids)
    env.contact_feet_ids = feet_cfg.body_ids
    # so im supposed to assign this; env.contact_joint_ids in init_contact_estimator before build_contact_estimator_frame() executed
    
    print("\n========== BODY NAMES ==========\n")
    for i, name in enumerate(env.scene["robot"].body_names):
        print(i, name)

    print("\n========== JOINT NAMES ==========\n")
    for i, name in enumerate(env.scene["robot"].joint_names):
        print(i, name)
    print("\n================================\n")
    
    joint_cfg = SceneEntityCfg("robot", joint_names=".*",)
    #joint_cfg = SceneEntityCfg("robot", joint_names=[".*HAA", ".*HFE", ".*KFE"])
    joint_cfg.resolve(env.scene)
    print("JOINT IDS:", joint_cfg.joint_ids)
    env.contact_joint_ids = joint_cfg.joint_ids  

    env.estimated_contacts = torch.zeros(
        (env.num_envs, 4),
        device=env.device,
        dtype=torch.float32,
    )

    #temporal buffer
    env.contact_history = torch.zeros(
        (env.num_envs,150,48),
        device=env.device,
        dtype=torch.float32,
    )
    print("[INFO] Contact estimator initialized.")
    print(f"[INFO] Num envs: {env.num_envs}")
    print(f"[INFO] Device: {env.device}")

def update_contact_estimator(env, env_ids=None):
    """
    Run contact estimator inference every step.
    """

    if not hasattr(env, "contact_estimator"):
        return
    
    frame = build_contact_estimator_frame(env)
    
    # buffer
    env.contact_history = torch.roll(env.contact_history,shifts=-1,dims=1)
    env.contact_history[:, -1] = frame

    contacts = env.contact_estimator.predict(env.contact_history)
    env.estimated_contacts = contacts.float()

    if env.common_step_counter % 100 == 0:

        est = env.estimated_contacts[0]

        print(
            "[EST] "
            f"LF={est[0]:.2f} | "
            f"RF={est[1]:.2f} | "
            f"LH={est[2]:.2f} | "
            f"RH={est[3]:.2f}"
        )

def reset_contact_estimator(env, env_ids=None):
    """
    Reset temporal buffers for selected envs.
    """
    if not hasattr(env, "contact_history"):
        return
    
    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
        )

    env.contact_history[env_ids] = 0.0
    env.estimated_contacts[env_ids] = 0.0
