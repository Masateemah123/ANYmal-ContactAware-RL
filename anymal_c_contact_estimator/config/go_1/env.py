# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils import configclass
from isaaclab_experiments.go1_low_level.configs.curriculum import CurriculumCfg
from isaaclab_experiments.go1_low_level.configs.recorder import (
    LowLevelRecorderManagerCfg,
)
from isaaclab_experiments.go1_low_level.configs.scene import MySceneCfg
from isaaclab_experiments.go1_low_level.configs.rewards import RewardsCfg
from isaaclab_experiments.go1_low_level.configs.observation import ObservationsCfg
from isaaclab_experiments.go1_low_level.configs.command import CommandsCfg, ActionsCfg
from isaaclab_experiments.go1_low_level.configs.termination import TerminationsCfg
from isaaclab_experiments.go1_low_level.configs.events import EventCfg

from isaaclab.sensors import (
    CameraCfg,
)
import isaaclab.sim as sim_utils
from isaaclab_experiments.go1_low_level.mdp.uniform_command import (
    UniformVelocityCommandCfg,
)

LOW_LEVEL_DT = 0.005  # 200HZ
LOW_LEVEL_DECIMATION = 4  # 5ms * 4 = 20ms control rate


@configclass
class UnitreeGo1RoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = LOW_LEVEL_DECIMATION
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = LOW_LEVEL_DT
        self.sim.render_interval = self.decimation

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    curriculum = None
    recorders = LowLevelRecorderManagerCfg()

    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.0, 5.0, 2.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="asset_body",
        env_index=0,
        resolution=(1920, 1080),
        asset_name="robot",
        body_name="trunk",
        cam_prim_path="/World/envs/env_0/Robot/trunk/front_cam",
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.viewer.resolution = (1920, 1080)
        # self.viewer.eye = (-2.0, 5.0, 2.0)
        # self.viewer.lookat = (0.0, 0.0, 0.0)
        # self.viewer.origin_type = "asset_body"
        # self.viewer.cam_prim_path = "/World/envs/env_0/Robot/trunk/front_cam"
        # self.viewer.asset_name = "robot"
        # self.viewer.body_name = "body"
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.episode_length_s = 5.0
        # remove random pushing event
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (10, 10)
        # self.commands.base_velocity.ranges = UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.0, 0.0),
        #     lin_vel_y=(-0.0, 0.0),
        #     ang_vel_z=(-0.0, 0.0),
        #     heading=(-0, 0),
        # )
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        #
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
            height=1080,
            width=1920,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=40.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-3, 0.0, 1),
                rot=(-0.589, -0.38, 0.389, 0.598),
                convention="opengl",
            ),
        )
