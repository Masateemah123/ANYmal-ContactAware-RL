# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_experiments.anymal_c_contact_estimator.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ROUGH_AND_RAMP_TERRAINS_CFG,
)
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip
from isaaclab_experiments.planning.configs.assets import UNITREE_GO1_CFG
import isaaclab_experiments.anymal_c_contact_estimator.mdp as mdp


# ─────────────────────────────────────────────────────────────
# R1 — Rough Baseline
# Pure proprioception, standard rewards, no contact estimator.
# ─────────────────────────────────────────────────────────────

@configclass
class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """R1 base config.
    Terrain stays on ROUGH_TERRAINS_CFG (set by LocomotionVelocityRoughEnvCfg).
    Do NOT touch terrain_generator here — R2/R3/R4 inherit this and must stay rough.
    """

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Disable raycaster / height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Pure proprioception — no base linear velocity
        #self.observations.policy.base_lin_vel = None


# ─────────────────────────────────────────────────────────────
# R2 — Rough + ContactNet Observation
# Adds estimated_contacts (4) to policy obs; rewards identical to R1.
# ─────────────────────────────────────────────────────────────

@configclass
class AnymalCRoughR2EnvCfg(AnymalCRoughEnvCfg):
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


# ─────────────────────────────────────────────────────────────
# R3 — Rough + Contact Consistency Reward (+0.05)
# R2 observations + contact_consistency reward; feet_air_time removed.
# ─────────────────────────────────────────────────────────────

@configclass
class AnymalCRoughR3EnvCfg(AnymalCRoughR2EnvCfg):
    """R3: R2 observations + contact_consistency as positive reward (+0.05).
    feet_air_time is zeroed out so gait emergence is driven by the estimator.
    """

    def __post_init__(self):
        super().__post_init__()

        # Remove feet_air_time (set weight to 0 — safer than deleting the term)
        self.rewards.feet_air_time.weight = 0.0

        # Add contact consistency as a reward
        self.rewards.contact_consistency = RewTerm(
            func=mdp.contact_estimator_consistency,
            weight=0.05,
        )


# ─────────────────────────────────────────────────────────────
# R4 — Rough + Contact Consistency Penalty (-0.05)
# Identical to R3 except the sign is flipped.
# ─────────────────────────────────────────────────────────────

@configclass
class AnymalCRoughR4EnvCfg(AnymalCRoughR2EnvCfg):
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
# R5 — Mixed Rough + Ramp Curriculum
# Terrain swap only — same observations/rewards as R1 (no contact estimator).
# Run against whichever of R1/R3/R4 wins, or in parallel if compute allows.
# ─────────────────────────────────────────────────────────────

@configclass
class AnymalCRoughRampEnvCfg(AnymalCRoughEnvCfg):
    """R5: identical to R1 except terrain is the mixed rough+ramp curriculum.
    Isolates terrain-curriculum effect — no reward or observation changes.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_AND_RAMP_TERRAINS_CFG


# ─────────────────────────────────────────────────────────────
# PLAY / EVAL variants  (never passed to train.py)
# ─────────────────────────────────────────────────────────────

def _apply_play_settings(cfg):
    """Shared helper: shrink scene, disable curriculum, disable noise/events."""
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
class AnymalCRoughEnvCfg_PLAY(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class AnymalCRoughR2EnvCfg_PLAY(AnymalCRoughR2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class AnymalCRoughR3EnvCfg_PLAY(AnymalCRoughR3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class AnymalCRoughR4EnvCfg_PLAY(AnymalCRoughR4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)


@configclass
class AnymalCRoughRampEnvCfg_PLAY(AnymalCRoughRampEnvCfg):
    """Held-out eval for R5. Eval on ramp-only or rough-only held-out terrain post-training."""

    def __post_init__(self):
        super().__post_init__()
        _apply_play_settings(self)