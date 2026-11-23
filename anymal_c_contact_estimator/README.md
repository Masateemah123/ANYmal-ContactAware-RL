ContactNet Dependency
=====================

This project depends on the ContactNet repository: https://github.com/EESC-LabRoM/contact-estimator-lars-2025

Clone: git clone https://github.com/EESC-LabRoM/contact-estimator-lars-2025.git

Expected checkpoint:noimu_lightningFINAL150L1-01-0.7734.ckpt

Update the path inside: contact_estimator_wrapper.py

or set: export CONTACTNET_CKPT=/path/to/checkpoint.ckpt before training.

## Objective

The primary objective is **velocity tracking performance** on rough terrain. The contact estimator is incorporated as an **additional observation** (and, in R3/R4, as a reward-shaping term) to evaluate whether proprioceptive contact belief information improves locomotion stability and terrain adaptability.

The policy is a **low-level locomotion policy**: it maps proprioceptive observations (plus, in R2–R4, estimated contact probabilities) directly to joint-level actions at a fixed control rate, conditioned on a commanded linear/angular velocity. It does not output footstep targets, gait parameters, or higher-level commands.

All experiments use the same PPO configuration, random seeds, command distribution, and training duration.

| Setting         | Value              |
| Training length | 200,000 iterations |


## Background

The contact estimator used as the PMETS observation/reward source is a CNN-GRU model trained offline on rosbag data of a Go1 quadruped walking across varied terrains. The trot gait is the predominant gait in the dataset, but other gaits were included to cover all 16 possible foot-contact states. Two checkpoints were saved (checkpoint 19 and checkpoint 27); checkpoint 27 is the one used here. Checkpoint 27 outputs softmax probabilities over contact states rather than binary contact labels, since binary states were judged too rigid for reward/observation shaping during RL training.

This estimator (trained on Go1) is now being used to inform an ANYmal C locomotion policy, both as an additional observation and as a reward/penalty term, to study gait emergence and trot stabilisation. A flat-terrain baseline has already been run, in which a trot gait emerged under both the reward and penalty formulations of the contact-consistency term (same weight magnitude, 0.05). This document specifies the rough-terrain campaign, which is where adaptability actually gets tested.

## Experiment R1 — Rough Baseline

**Purpose:** Reference locomotion policy trained using the standard IsaacLab reward structure.

**Observations:**
- Standard proprioceptive observations

**Rewards:**

| Term | Weight |
|---|---|
| `track_lin_vel_xy_exp` | 1.0 |
| `track_ang_vel_z_exp` | 0.5 |
| `lin_vel_z_l2` | -2.0 |
| `ang_vel_xy_l2` | -0.05 |
| `dof_torques_l2` | -1.0e-5 |
| `dof_acc_l2` | -2.5e-7 |
| `action_rate_l2` | -0.01 |
| `feet_air_time` | 0.5 |
| `flat_orientation_l2` | -5.0 |
| `undesired_contacts` | -1.0 |

## Experiment R2 — Rough + ContactNet Observation

**Purpose:** Evaluate transferability of ContactNet output as an additional observation source.

**Observations:**
- Standard observations
- `estimated_contacts` (4)

**Rewards:**
- Identical to R1
- `feet_air_time` = 0.5
- No Contact reward term


## Experiment R3 — Rough + Contact Consistency Reward

**Purpose:** Evaluate Contact reward shaping using contact agreement.

**Observations:**
- Standard observations
- `estimated_contacts` (4)

**Rewards:**
- Same as R2, with the following changes:

| Change | Term | Weight |
|---|---|---|
| Remove | `feet_air_time` | 0.0 |
| Add | `contact_consistency` | +0.05 |


## Experiment R4 — Rough + Contact Consistency Penalty

**Purpose:** Evaluate Contact penalty formulation.

**Observations:**
- Standard observations
- `estimated_contacts` (4)

**Rewards:**
- Same as R2, with the following changes:

| Change | Term | Weight |
|---|---|---|
| Remove | `feet_air_time` | 0.0 |
| Add | `contact_consistency` | -0.05 |


## Experiment R5 — Mixed Rough + Ramp Curriculum (Deployment Generalization)

**Purpose:** Evaluate whether a policy trained on a mixed rough/slope curriculum transfers better to ramp deployment than a rough-only policy. Rough terrain trains the policy against unpredictable, high-frequency height perturbations; ramps are a smooth, low-frequency, sustained pitch disturbance. The two do not fully transfer, so this experiment is needed to support any ramp-deployment claim.

**Observations:**
- Same as the corresponding R1/R3/R4 variant (run once contact-shaping sign is decided from R1–R4 results, or run in parallel if compute allows)

**Terrain:**
- Mixed curriculum: ~50% random rough, ~25% upward pyramid slope, ~25% downward (inverted pyramid) slope
- Curriculum progresses slope steepness and roughness together across difficulty rows, consistent with the existing rough-terrain curriculum setup
- A held-out **ramp-only** evaluation terrain and a held-out **rough-only** evaluation terrain are kept separate from the mixed training terrain, so transfer can be measured cleanly post-training (i.e., evaluate the mixed-curriculum policy on ramp-only terrain it didn't directly train the bulk of its samples on)

**Rewards:**
- Identical to whichever of R1/R3/R4 this is run against (no reward changes — this experiment isolates the effect of terrain curriculum, not reward shaping)

**Rationale:**
- Determines which training condition (rough-only vs. mixed rough+ramp) actually matters for a successful ramp deployment, rather than assuming rough-terrain training generalizes to slopes by default.


### Training Metrics
- `Train/mean_reward`
- `Train/mean_episode_length`
- `Loss/value_function`
- `Loss/surrogate`
- `Policy/mean_noise_std`

### Velocity Tracking
- `track_lin_vel_xy_exp`
- `track_ang_vel_z_exp`
- `Metrics/error_vel_xy`
- `Metrics/error_vel_yaw`

### Stability
- `lin_vel_z_l2`
- `ang_vel_xy_l2`
- `flat_orientation_l2`
- Episode Length
- Fall Count

### Energy
- `dof_torques_l2`
- `dof_acc_l2`
- `action_rate_l2`
- Average Joint Torque
- LF Torque
- RF Torque
- LH Torque
- RH Torque
- Cost of Transport (CoT)

### Contact Metrics
- `contact_consistency`
- `contact_confidence`
- Contact Agreement Score (CAS)

### Gait Regularity Metrics

These are computed from **ground-truth contact state** (simulator contact sensors), independent of the contact estimator, so they validate trot stabilisation rather than estimator agreement.

| Metric | Definition |
|---|---|
| `Gait/stride_time_cv` | Coefficient of variation of stride time, per foot, averaged across feet. Lower = more rhythmic. |
| `Gait/duty_factor_mean` | Mean stance-phase fraction per foot across the episode. Should sit near 0.5 for a trot. |
| `Gait/duty_factor_std` | Standard deviation of duty factor across the episode. Lower = more consistent stance timing. |
| `Gait/diagonal_phase_offset` | Circular distance of diagonal leg-pair (LF–RH, RF–LH) phase from 180°. Lower = closer to ideal trot phasing. |
| `Gait/contact_pattern_entropy` | Entropy of the 16-state ground-truth contact pattern distribution over an episode. Lower = more periodic/structured gait. |

### Post-Processing Figures
- Contact Raster Plot
- Ground Truth Contact Timeline
- Estimated Contact Timeline
- Contact Agreement Plot
- Velocity Tracking Plot
- Cost of Transport Plot
- Torque Distribution Plot
- Episode Length Distribution
- Fall Rate Comparison
- Stride Time Variance Plot (per foot)
- Duty Factor Distribution Plot
- Ramp vs. Rough Transfer Comparison (R5 evaluated on held-out ramp-only and rough-only terrains)

---

## Implementation Handoff — R5

This section is the exact, file-level set of changes needed to add R5 to the existing codebase. No other experiment (R1–R4) is affected by these changes.

### File 1: `velocity_env_cfg.py`

Add the terrain generator imports and the new mixed-curriculum config. This file already defines `ROUGH_TERRAINS_CFG`'s import and `MySceneCfg`; **do not change** `MySceneCfg.terrain.terrain_generator` — it must keep pointing at `ROUGH_TERRAINS_CFG` so R1–R4 stay on rough-only terrain.

Add near the top, alongside the existing terrain import:

```python
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # used by R1-R4, unchanged
```

Add this new config (e.g. right after the imports, before `MySceneCfg`):

```python
# ─────────────────────────────────────────────────────────────
# R5 terrain: mixed rough + ramp curriculum
# ─────────────────────────────────────────────────────────────
ROUGH_AND_RAMP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.25,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
```

**Notes:**
- `proportion` values must sum to 1.0.
- `curriculum=True` means slope steepness and roughness both progress together across difficulty rows — same progressive-difficulty behavior as the existing rough-only curriculum.
- If ramps render as vertical walls or disappear, raise `slope_threshold`.

---

### File 2: `rough_env_cfg.py`

Two structural fixes versus a first-draft version of this file:

1. **Class order matters.** `AnymalCRoughRampEnvCfg` inherits from `AnymalCRoughEnvCfg`, so `AnymalCRoughEnvCfg` must be defined *first* in the file — Python resolves names top-to-bottom, and defining the subclass before its parent raises `NameError`.
2. **The terrain swap must live only in the R5 subclass, not in the shared base class.** Setting `self.scene.terrain.terrain_generator = ROUGH_AND_RAMP_TERRAINS_CFG` inside `AnymalCRoughEnvCfg` would silently put R1, R2, R3, R4, *and* the rough `_PLAY` config all onto ramp terrain — breaking the controlled comparison the whole campaign depends on.

Corrected file:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_experiments.anymal_c_contact_estimator.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ROUGH_AND_RAMP_TERRAINS_CFG,
)
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
import isaaclab_experiments.anymal_c_contact_estimator.mdp as mdp


@configclass
class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """R1-R4 base config. Terrain stays whatever LocomotionVelocityRoughEnvCfg
    set it to (ROUGH_TERRAINS_CFG) — do NOT set terrain_generator here, or
    R1-R4 will silently train on ramps too."""

    def __post_init__(self):
        super().__post_init__()
        # switch robot to anymal-c
        # self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- disable raycaster ---
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # --- pure proprioception ---
        # remove base linear velocity
        self.observations.policy.base_lin_vel = None

        # # ----------------------------------------------------
        # # CONTACT ESTIMATOR OUTPUT -> PPO OBSERVATION
        # # ----------------------------------------------------
        # # Uncomment for R2 / R3 / R4 (estimated_contacts must be in the
        # # observation space for the contact-consistency reward to mean anything)
        # self.observations.policy.estimated_contacts = ObsTerm(
        #     func=mdp.estimated_contacts,
        # )
        # # ----------------------------------------------------
        # # CONTACT ESTIMATOR EVENTS
        # # ----------------------------------------------------
        # self.events.init_contact_estimator = EventTerm(
        #     func=mdp.init_contact_estimator,
        #     mode="startup",
        # )
        # self.events.update_contact_estimator = EventTerm(
        #     func=mdp.update_contact_estimator,
        #     mode="interval",
        #     interval_range_s=(0.005, 0.005),  # = sim.dt
        # )
        # self.events.reset_contact_estimator = EventTerm(
        #     func=mdp.reset_contact_estimator,
        #     mode="reset",
        # )


@configclass
class AnymalCRoughRampEnvCfg(AnymalCRoughEnvCfg):
    """R5 config: identical to AnymalCRoughEnvCfg (same robot, same
    proprioception setup, same observation/event toggles above) except
    for terrain, which is swapped to the mixed rough+ramp curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_AND_RAMP_TERRAINS_CFG


@configclass
class AnymalCRoughEnvCfg_PLAY(AnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # disable observation noise for play
        self.observations.policy.enable_corruption = False
        # disable disturbance events for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalCRoughRampEnvCfg_PLAY(AnymalCRoughRampEnvCfg):
    """Held-out eval variant for R5 — same terrain swap, same play-mode
    simplifications as AnymalCRoughEnvCfg_PLAY. Eval/visualization only —
    never used for training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

**Reminder for R2/R3/R4 specifically:** the `estimated_contacts` observation term and the three contact-estimator events inside `AnymalCRoughEnvCfg.__post_init__` are currently commented out. They must be uncommented for R2, R3, and R4 (otherwise those runs train with no contact information reaching the policy at all, despite the reward/observation spec calling for it). R1 and R5 can leave them commented out, since R5 isolates the terrain-curriculum effect only and doesn't change observations or rewards.

---

### File 3: `__init__.py` (gym task registration)

Add two new `gym.register` blocks for R5 — a training task and a play/eval task — following the exact pattern already used for the rough and flat tasks:

```python
# ─────────────────────────────────────────────────────────────
# R5 — Rough + Ramp mixed-curriculum (deployment generalization)
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Ramp-Anymal-C-Contact-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughRampEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Ramp-Anymal-C-Contact-Estimator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughRampEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)
```

Insert these blocks anywhere after the existing rough-task registrations (ordering relative to the flat-task blocks doesn't matter). R5 reuses `AnymalCRoughPPORunnerCfg` for PPO hyperparameters since R5 changes terrain only, not rewards — this keeps R5 a clean isolation of the terrain-curriculum effect. A dedicated PPO runner config for R5 can be added later in `agents/rsl_rl_ppo_cfg.py` if needed.

---

### Launching R5

**Training** (this is the actual training run — 2048 envs, full curriculum, 200,000 iterations, same as R1–R4):

```bash
python train.py --task Isaac-Velocity-Rough-Ramp-Anymal-C-Contact-Estimator-v0
```

**Evaluation/visualization only**, after training completes (50 envs, no curriculum, no noise — this is for watching/measuring the trained policy, not for training it):

```bash
python play.py --task Isaac-Velocity-Rough-Ramp-Anymal-C-Contact-Estimator-Play-v0
```

`AnymalCRoughRampEnvCfg_PLAY` is never passed to `train.py`. Training always targets the non-`_PLAY` config.