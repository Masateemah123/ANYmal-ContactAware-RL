# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause
"""Play/evaluate a trained Anymal-C locomotion policy with diagnostics."""

import argparse
from isaaclab.app import AppLauncher

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play a trained Anymal-C policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Anymal-C-Contact-Estimator-v0",
)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# ── launch simulator ───────────────────────────────────────────────────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after sim launch ───────────────────────────────────────────────
import os
import time
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks        # noqa: F401
import isaaclab_experiments  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg, load_cfg_from_registry


# ───────────────────────────────────────────────────────────────────────────
def main():
    # ── ENV CONFIG ────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # clean evaluation setup
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum = None

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.max_init_terrain_level = None

    # ── AGENT CONFIG ──────────────────────────────────────────────────────
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
        args_cli.task, "rsl_rl_cfg_entry_point"
    )

    # ── CHECKPOINT ────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Loading experiment from: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # ── ENV CREATION ──────────────────────────────────────────────────────
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ── VIDEO ─────────────────────────────────────────────────────────────
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ── RSL-RL WRAPPER ────────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)

    # ── LOAD POLICY ───────────────────────────────────────────────────────
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ── EXPORT (optional) ─────────────────────────────────────────────────
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    normalizer = getattr(ppo_runner, "obs_normalizer", None)
    export_dir = os.path.join(log_dir, "exported")

    try:
        export_policy_as_jit(policy_nn, normalizer, export_dir, "policy.pt")
        export_policy_as_onnx(policy_nn, normalizer, export_dir, "policy.onnx")
        print(f"[INFO] Exported policy to: {export_dir}")
    except Exception as e:
        print(f"[WARN] Export skipped: {e}")

    # ── ROLLOUT LOOP ──────────────────────────────────────────────────────
    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    print("[INFO] Starting playback with diagnostics.")

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # ── DEBUG PRINTS ──────────────────────────────────────────────
            if timestep % 100 == 0:
                robot = env.unwrapped.scene["robot"]

                # COMMAND
                cmd = env.unwrapped.command_manager.get_command("base_velocity")

                # ACTUAL VELOCITY
                vel = robot.data.root_lin_vel_b

                # ORIENTATION (gravity projection)
                gravity = robot.data.projected_gravity_b

                # HEIGHT
                height = robot.data.root_pos_w[:, 2]

                print("\n================ DEBUG =================")
                print(f"[CMD ] vx={cmd[0,0]:+.3f} vy={cmd[0,1]:+.3f} yaw={cmd[0,2]:+.3f}")
                print(f"[ACT ] vx={vel[0,0]:+.3f} vy={vel[0,1]:+.3f}")
                print(f"[ERR ] vx_err={(cmd[0,0]-vel[0,0]):+.3f}")
                print(f"[HEIGHT] {height[0]:.3f}")
                print(f"[GRAV ] roll/pitch proxy = {gravity[0,:2]}")
                print("========================================\n")

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        timestep += 1

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
    simulation_app.close()
