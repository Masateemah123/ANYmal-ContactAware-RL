# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .rough_env_cfg import AnymalDRoughEnvCfg
# from isaaclab_experiments.anymal_c_contact_estimator.velocity_env_cfg import LocomotionVelocityFlatEnvCfg
# from isaaclab_experiments.anymal_c_contact_estimator.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
#from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

import isaaclab_experiments.anymal_c_contact_estimator.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm

@configclass
class AnymalDFlatEnvCfg(AnymalDRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        print("CONTACTNET FLAT CFG LOADED")
        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        #self.rewards.feet_air_time.weight = 0.5ss

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Remove terrain sensing
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Disable terrain curriculum
        self.curriculum.terrain_levels = None

        
        # ----------------------------------------------------
        # CONTACT ESTIMATOR OUTPUT -> PPO OBSERVATION
        # ----------------------------------------------------
        # I temporally disable this to get my old checkpoints to work 
        self.observations.policy.estimated_contacts = ObsTerm(
            func=mdp.estimated_contacts,
        )

        # # ----------------------------------------------------
        # # CONTACT ESTIMATOR EVENTS
        # # ----------------------------------------------------

        self.events.init_contact_estimator = EventTerm(
            func=mdp.init_contact_estimator,
            mode="startup",
        )

        self.events.update_contact_estimator = EventTerm(
            func=mdp.update_contact_estimator,
            mode="interval",
            interval_range_s=(0.005, 0.005),  # = sim.dt
        )

        self.events.reset_contact_estimator = EventTerm(
            func=mdp.reset_contact_estimator,
            mode="reset",
        )

@configclass
class AnymalDFlatEnvCfg_PLAY(AnymalDFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        