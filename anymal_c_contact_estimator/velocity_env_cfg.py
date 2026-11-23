# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise # inject uniform distribution noise into observations
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # for R1–R4

from . import mdp
from .mdp.contact_estimator_event import (
    init_contact_estimator,
    update_contact_estimator,
    reset_contact_estimator,
)

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
#from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
#from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  
from .mdp.observations import estimated_contacts
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

# ─────────────────────────────────────────────────────────────
# R5 terrain: mixed rough + ramp curriculum
# ─────────────────────────────────────────────────────────────
ROUGH_AND_RAMP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True, #all the terrain proportions sum to 1. 
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        
    },
)

RAMP_TESTING_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True, #all the terrain proportions sumto 1. 
    sub_terrains={
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            #slope_range=(0.10, 0.10), #5%
            #slope_range=(0.18, 0.18), #10%
            #slope_range=(0.27,0.27), #(15)
            #slope_range=(0.36,0.36), #20%
            slope_range=(0.45,0.45), #25%
            #slope_range=(0.54,0.54), #30%
            #slope_range=(0.63,0.63), #35%
            #slope_range=(0.75, 0.75), 
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.45,0.45), #5%
            platform_width=2.0,
            border_width=0.25,
        ),
        
    },
)
# ─────────────────────────────────────────────────────────────
# Scene
# ─────────────────────────────────────────────────────────────
@configclass
class MySceneCfg(InteractiveSceneCfg):
    
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        #terrain_generator=ROUGH_TERRAINS_CFG,# ← swap this for R5
        terrain_generator = ROUGH_AND_RAMP_TERRAINS_CFG,
        max_init_terrain_level=3, #was 5 
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    #robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    #robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    #sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner = None
    
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    #lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    ) 
##
# MDP settings
##     

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
        # this is for when we are doing the ramp test dont forget!!!
        #ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #    lin_vel_x=(0.5,0.5),lin_vel_y=(0.0,0.0),ang_vel_z=(0.0,0.0),heading=(0.0,0.0),
        #    ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    
    """Observation specifications for the MDP."""
    # def estimated_contacts_obs(env):
    #     return env.estimated_contacts

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1), 
        #     clip=(-1.0, 1.0),
        # )

        estimated_contacts = ObsTerm(func=estimated_contacts) # where is this estimated contacts defined

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# ─────────────────────────────────────────────────────────────
# Events (Estimator lives here)
# ─────────────────────────────────────────────────────────────
@configclass
class EventCfg:
    
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    init_contact_estimator = EventTerm(
        func=init_contact_estimator,
        mode="startup",
    )

    reset_contact_estimator = EventTerm(
        func=reset_contact_estimator,
        mode="reset",
    )

    update_contact_estimator = EventTerm(
        func=update_contact_estimator,
        mode="interval",
        interval_range_s=(0.005, 0.005), # this should be equal to sim.dt to update every step, was (0.0, 0.0)
    )
    
    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

# Rewards
# ─────────────────────────────────────────────────────────────

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # feet_stance_width = RewTerm(
    # func=mdp.feet_stance_width,
    # weight=-1.0,
    # params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True),
    #     "min_distance": 0.1,
    # },
    # )
    
    hip_joint_position = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        }
    )

    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # -- estimator; Contact Estimation Consistency
    # Why does a reward that only compares estimated contacts to ground-truth contacts induce a gait structure?
    # Contact-state consistency acts as a self-organizing principle for locomotion. By rewarding agreement between internal 
    # contact beliefs and external contact reality, periodic quadrupedal gaits emerge without explicit gait-shaping rewards.
    contact_consistency = RewTerm(
        func=mdp.contact_estimator_consistency,
        weight=0.5 
    )
    
    # contact_confidence = RewTerm(
    # func=mdp.contact_confidence_reward,
    # weight=0.05,
    # )
    
    # pmets_fqf = RewTerm(
    # func=mdp.pmets_fqf_reward,
    # weight=1.0,
    # )

    # -- contact estimator guided stability
    # contact_confidence = RewTerm(
    #     func=mdp.contact_confidence_reward,
    #     weight=0.4,
    # )
    # marginal_support = RewTerm(
    #     func=mdp.marginal_support_penalty,
    #     weight=0.6,
    # )
    # terrain_uncertainty = RewTerm(
    #     func=mdp.terrain_uncertainty_penalty,
    #     weight=0.5,
    # )
    # foothold_quality = RewTerm(
    #     func=mdp.foothold_quality_reward,
    #     weight=0.8,
    # )

    # # -- remove or reduce contact_consistency
    # # Now replaced by the above four terms which are more precise
    # # contact_consistency = RewTerm(weight=0.1)  # keep very small or remove
        
    # THIS IS FOR THE PMETS INTEGRATION MODE B (using the grey-score as a penalty)
    # pmets_grey_scale = RewTerm(
    # func=mdp.pmets_grey_scale_penalty,
    # weight=0.05,
    # )

# ─────────────────────────────────────────────────────────────
# Termination / Curriculum
# ─────────────────────────────────────────────────────────────

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #time_out = None
    
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        #params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────

@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    # scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4 #200/4 = 50 Hz policy update rate, which is good for the contact estimator and also not too slow for learning
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005 #= 1 / 0.005 = 200 Hz physics update rate, which is good for the contact estimator
        # sim.dt = 0.001 and decimation=1 to keep up with the contact estimator..
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # bad orientation
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2}) 
        # reward standing height
        self.rewards.base_height = RewTerm(func=mdp.base_height_l2,
                                        weight=3.0,  # positive — try 2.0-5.0 and tune
                                        params={"target_height": 0.30},  # confirm via the zero-action measurement
        )

        # make sure trunk contact actually terminates
        #self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        # hard floor: end the episode if the body drops too low (kills the crawl)
        self.terminations.base_too_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.18},
        )
        
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.25
                    
        

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # history horizon = 150/1000 = 0.15 s = 150 ms 150/50=3.0 s. You are feeding ContactNet:3 seconds when it was trained on 150 milliseconds. That is a 20× mismatch.
        # To preserve the original horizon:0.15 s × 50 Hz=7.5 samples
        
@configclass
class LocomotionTestEnvCfg(LocomotionVelocityRoughEnvCfg):

    def __post_init__(self):
        # general settings
        self.scene.num_envs = 2
        self.scene.terrain.terrain_generator = RAMP_TESTING_TERRAINS_CFG
        self.decimation = 4 #200/4 = 50 Hz policy update rate, which is good for the contact estimator and also not too slow for learning
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005 #= 1 / 0.005 = 200 Hz physics update rate, which is good for the contact estimator
        # sim.dt = 0.001 and decimation=1 to keep up with the contact estimator..
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt