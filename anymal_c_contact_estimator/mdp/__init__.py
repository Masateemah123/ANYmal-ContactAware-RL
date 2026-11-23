# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculums import *  # noqa: F401, F403
#from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .contact_estimator_event import init_contact_estimator, update_contact_estimator, reset_contact_estimator  # noqa: F401
from .observations import estimated_contacts
from .rewards import (
    contact_estimator_consistency, feet_stance_width, stand_still_joint_deviation_l1, 
    track_ang_vel_z_world_exp, track_lin_vel_xy_yaw_frame_exp, feet_slide, 
    feet_air_time_positive_biped, feet_air_time
)

# Isaac-Velocity-Rough-Anymal-R1-v0
# Isaac-Velocity-Rough-Anymal-R2-v0
# Isaac-Velocity-Rough-Anymal-R3-v0
# Isaac-Velocity-Rough-Anymal-R4-v0


