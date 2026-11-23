# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""This file is ONLY for task registration."""

import gymnasium as gym

from isaaclab_experiments.anymal_c_contact_estimator.config.anymal_c import agents as anymal_c_agents  # noqa: F401
from isaaclab_experiments.anymal_c_contact_estimator.config.go_1 import agents as go1_agents  # noqa: F401


##
# Register Gym environments.
##

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — R1 Rough Baseline
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Contact-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Contact-Estimator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — R2 Rough + ContactNet Observation
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Contact-Estimator-R2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR2EnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Contact-Estimator-R2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — R3 Rough + Contact Consistency Reward (+0.05)
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Contact-Estimator-R3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR3EnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-D-Contact-Estimator-R3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — R4 Rough + Contact Consistency Penalty (-0.05)
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Contact-Estimator-R4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR4EnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-Contact-Estimator-R4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughR4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — R5 Rough + Ramp mixed-curriculum
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Ramp-Anymal-C-Contact-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughRampEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
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
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — Flat
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Contact-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Contact-Estimator-With-Symmetry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-Contact-Estimator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.flat_env_cfg:AnymalCFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# ANYMAL-C — Alternate R1-R4 aliases
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Isaac-Velocity-Rough-Anymal-R1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-R2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughContactObsEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-R3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughContactRewardEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-R4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.config.anymal_c.rough_env_cfg:AnymalCRoughContactPenaltyEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Anymal-RampTest-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.velocity_env_cfg:LocomotionTestEnvCfg",
        "rsl_rl_cfg_entry_point":
            f"{anymal_c_agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Rough R1 baseline
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Rough R2 ContactNet observation
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Rough R3 ContactNet obs + consistency reward (+0.05)
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Rough R6 consistency reward ALONE (no ContactNet observation)
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Rough-R6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.rough_env_cfg:UnitreeGo1RoughR6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Flat R1 baseline
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.flat_env_cfg:UnitreeGo1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.flat_env_cfg:UnitreeGo1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)

# ─────────────────────────────────────────────────────────────
# GO1 — Flat R2 ContactNet observation
# ─────────────────────────────────────────────────────────────

gym.register(
    id="Go1-LowLevel-Unitree-Flat-R2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.flat_env_cfg:UnitreeGo1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go1-LowLevel-Unitree-Flat-R2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.config.go_1.flat_env_cfg:UnitreeGo1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{go1_agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1FlatPPORunnerCfg",
    },
)