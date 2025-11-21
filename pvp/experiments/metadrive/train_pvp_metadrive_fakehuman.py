#!/usr/bin/env python3
import argparse
import os
import uuid
from pathlib import Path

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.utils.waypoint_viz import drop_waypoint_markers, clear_waypoint_markers

from panda3d.core import (
    TextNode,
    LineSegs,
    NodePath,
    Vec4,
    TransparencyAttrib,
    ColorAttrib,
)

# If your local copy needs DummyVecEnv anywhere, import it. (Some variants of this file did.)
# from pvp.sb3.common.vec_env import DummyVecEnv


def clear_waypoint_markers(env):
    """Remove previously created waypoint markers, line, and debug parent if any."""
    try:
        # remove child markers
        if hasattr(env, "_wp_markers"):
            for np in env._wp_markers:
                try:
                    np.removeNode()
                except Exception:
                    pass
            env._wp_markers = []
        # remove line
        if hasattr(env, "_wp_line") and env._wp_line is not None:
            try:
                env._wp_line.removeNode()
            except Exception:
                pass
            env._wp_line = None
        # remove debug parent (so we also drop any accumulated render state)
        if hasattr(env, "_wp_debug_np") and env._wp_debug_np is not None and not env._wp_debug_np.isEmpty():
            try:
                env._wp_debug_np.removeNode()
            except Exception:
                pass
            env._wp_debug_np = None
    except Exception:
        pass


def _ensure_debug_parent(env):
    """
    Make (or recreate) a dedicated unshaded parent NodePath that forces an
    unlit, untextured, flat-colored pipeline for all children.
    """
    if not hasattr(env, "_wp_debug_np") or env._wp_debug_np is None or env._wp_debug_np.isEmpty():
        parent = env.engine.render.attachNewNode("debug_waypoints")
        # Opt out of shader-based pipelines and lighting/material/texture so colors are flat.
        parent.setShaderOff(1)
        parent.setLightOff(1)
        parent.setTextureOff(1)
        parent.setMaterialOff(1)
        # Block any inherited ColorScale (global dimming).
        try:
            parent.setColorScaleOff(1)
        except Exception:
            parent.clearColorScale()
        parent.setTransparency(TransparencyAttrib.M_alpha)
        env._wp_debug_np = parent
    return env._wp_debug_np




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="pvp_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="log", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_human_proxy_value_loss", default="True", type=str)
    parser.add_argument("--with_agent_proxy_value_loss", default="True", type=str)
    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    free_level = args.free_level

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(
            # Open the interface
            use_render=True,
            # FakeHumanEnv config:
            free_level=free_level,
            save_screenshots=True,
            vlm_wait_time=3.0,
            screenshot_dir="ima_log",
            screenshot_prefix="shot",

            
        ),

        # Algorithm config
        algo=dict(
            adaptive_batch_size=args.adaptive_batch_size,
            bc_loss_weight=args.bc_loss_weight,
            only_bc_loss=args.only_bc_loss,
            with_human_proxy_value_loss=args.with_human_proxy_value_loss,
            with_agent_proxy_value_loss=args.with_agent_proxy_value_loss,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # reward-free!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # < 50K steps
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Also build the eval env =====
    # MOVED this block BEFORE creating the training env to avoid engine re-initialization in subprocesses.
    def _make_eval_env():
        eval_env_config = dict(
            manual_control=False,
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor as _Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = _Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = SubprocVecEnv([_make_eval_env])

    # ===== Setup the training environment =====
    train_env = FakeHumanEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))

    # ===== Drop debug waypoints =====
    train_env.reset()  # make sure the environment has started
    nav = getattr(train_env.agent, "navigation", None)
    waypoints = getattr(nav, "target_checkpoints", []) if nav is not None else []
    coords = []
    for i, wp in enumerate(waypoints):
        pos = getattr(wp, "position", None) or getattr(wp, "pos", None)
        if pos is not None:
            if len(pos) >= 3:
                coords.append((float(pos[0]), float(pos[1]), float(pos[2])))
            else:
                coords.append((float(pos[0]), float(pos[1]), 0.0))
    print("Waypoint coordinates:", coords)

    # Example manual waypoints
    manual_waypoints = [
        (0.0, 0.0, 0.0),       # near start
        (10.0, 0.0, 0.0),      # 10m forward
        (20.0, 5.0, 0.0),      # a little right
        (30.0, 5.0, 0.0),      # 10m more forward
        (40.0, 0.0, 0.0),      # back toward center line
    ]

    # Drop your markers (choose any RGBA you like)
    # drop_waypoint_markers(train_env, manual_waypoints, color=(0.0, 0.0, 1.0, 1.0))
    drop_waypoint_markers(train_env, manual_waypoints, color=(1.0, 0.0, 0.0, 1.0))

    # Store all shared control data to the files.
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    save_freq = args.save_freq
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PVPTD3(**config["algo"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,


        # eval_env=eval_env,
        # eval_freq=150,
        # n_eval_episodes=50,
        # eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )