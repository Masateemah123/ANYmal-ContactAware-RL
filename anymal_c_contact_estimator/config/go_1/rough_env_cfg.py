# config/go_1/rough_env_cfg.py

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_experiments.anymal_c_contact_estimator.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ROUGH_AND_RAMP_TERRAINS_CFG,
)
import isaaclab_experiments.anymal_c_contact_estimator.mdp as mdp

# use whichever Go1 asset cfg you've confirmed is correct — you referenced two
# different sources earlier (isaaclab_assets.robots.unitree.UNITREE_GO1_CFG vs
# isaaclab_experiments.planning.configs.assets.UNITREE_GO1_CFG). Pick one:
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG


# ─────────────────────────────────────────────────────────────
# R1 — Rough Baseline (Go1)
# ─────────────────────────────────────────────────────────────

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # fix ANYmal-naming body patterns for Go1's naming convention —
        # VERIFY these against your actual USD before trusting them (see earlier message)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_com.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"

@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)

# ─────────────────────────────────────────────────────────────
# R2 — Rough + ContactNet Observation
# Adds estimated_contacts (4) to policy obs; rewards identical to R1.
# ─────────────────────────────────────────────────────────────

@configclass
class UnitreeGo1RoughR2EnvCfg(UnitreeGo1RoughEnvCfg):
    """R2: standard rewards + ContactNet output as additional observation."""
    def __post_init__(self):
        super().__post_init__()
        # ContactNet output → policy observation
        self.observations.policy.estimated_contacts = ObsTerm(
            func=mdp.estimated_contacts,
        )
        # ContactNet lifecycle events
        self.events.init_contact_estimator = EventTerm(
            func=mdp.init_contact_estimator,
            mode="startup",
        )
        self.events.update_contact_estimator = EventTerm(
            func=mdp.update_contact_estimator,
            mode="interval",
            interval_range_s=(0.005, 0.005),  # matches sim.dt = 0.005 s
        )
        self.events.reset_contact_estimator = EventTerm(
            func=mdp.reset_contact_estimator,
            mode="reset",
        )

@configclass
class UnitreeGo1RoughR2EnvCfg_PLAY(UnitreeGo1RoughR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)

# ─────────────────────────────────────────────────────────────
# R3 — Rough + ContactNet Obs + Consistency Reward (+0.05)
# ─────────────────────────────────────────────────────────────

@configclass
class UnitreeGo1RoughR3EnvCfg(UnitreeGo1RoughR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.contact_consistency = RewTerm(func=mdp.contact_estimator_consistency, weight=0.05)


@configclass
class UnitreeGo1RoughR3EnvCfg_PLAY(UnitreeGo1RoughR3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)

# ─────────────────────────────────────────────────────────────
# R4 — Rough + ContactNet Obs + Consistency Penalty (-0.05)
# ─────────────────────────────────────────────────────────────

@configclass
class UnitreeGo1RoughR4EnvCfg(UnitreeGo1RoughR2EnvCfg):
    """R4: R2 observations + contact_consistency as penalty (-0.05).
    feet_air_time is zeroed out — same as R3.
    """

    def __post_init__(self):
        super().__post_init__()

        # Remove feet_air_time
        self.rewards.feet_air_time.weight = 0.0

        # Add contact consistency as a penalty
        self.rewards.contact_consistency = RewTerm(
            func=mdp.contact_estimator_consistency,
            weight=-0.05,
        )

# ─────────────────────────────────────────────────────────────
# R6 — Contact Consistency Reward ALONE (no ContactNet observation)
# ─────────────────────────────────────────────────────────────

@configclass
class UnitreeGo1RoughR6EnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.init_contact_estimator = EventTerm(func=mdp.init_contact_estimator, mode="startup")
        self.events.update_contact_estimator = EventTerm(
            func=mdp.update_contact_estimator, mode="interval", interval_range_s=(0.005, 0.005),
        )
        self.events.reset_contact_estimator = EventTerm(func=mdp.reset_contact_estimator, mode="reset")
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.contact_consistency = RewTerm(func=mdp.contact_estimator_consistency, weight=0.05)


@configclass
class UnitreeGo1RoughR6EnvCfg_PLAY(UnitreeGo1RoughR6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


# ─────────────────────────────────────────────────────────────
# PLAY helper 
# ─────────────────────────────────────────────────────────────

def _apply_play_settings(cfg):
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    
@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class UnitreeGo1RoughR2EnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class UnitreeGo1RoughR3EnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class UnitreeGo1RoughR4EnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class UnitreeGo1RoughRampEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    """Held-out eval for R5. Eval on ramp-only or rough-only held-out terrain post-training."""

    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)
