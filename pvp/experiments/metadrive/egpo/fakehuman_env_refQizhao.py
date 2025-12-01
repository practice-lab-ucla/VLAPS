import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.policy.env_input_policy import EnvInputPolicy

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv

from VLM import run_vlm

import time   


FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()


def get_expert():
    from pvp.sb3.common.save_util import load_from_zip_file
    from pvp.sb3.ppo import PPO
    from pvp.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})

    # Initialize agent
    algo_config = dict(

        
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metadrive_pvp_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2 * math.pi * var)**.5
    num = math.exp(-(float(x) - float(mean))**2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert = get_expert()


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None

    def __init__(self, config):
        super(FakeHumanEnv, self).__init__(config)

        
        self._vlm_pending = False
        self._vlm_wait_until = 0.0


        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins**2)
        else:
            return super(FakeHumanEnv, self).action_space

    # def _preprocess_actions(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray], int]) -> Union[np.ndarray, Dict[AnyStr, np.ndarray], int]:
    #     if self.config["use_discrete"]:
    #         print(111)
    #         return int(actions)
    #     else:
    #         return actions

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(FakeHumanEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "free_level": 0.95,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,

                # add these defaults so training can override safely
                "save_screenshots": False,
                "vlm_wait_time": 3.0,
                "screenshot_dir": "ima_log",
                "screenshot_prefix": "shot",
            },
            allow_add_new_key=True,   # <— important
        )
        return config

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        # ===== Get expert action and determine whether to take over! =====

        if self.config["disable_expert"]:
            pass

        else:
            if self.expert is None:
                global _expert
                self.expert = _expert
            last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
            distribution = self.expert.get_distribution(last_obs)
            log_prob = distribution.log_prob(torch.from_numpy(actions).to(last_obs.device))
            action_prob = log_prob.exp().detach().cpu().numpy()

            if self.config["expert_deterministic"]:
                expert_action = distribution.mode().detach().cpu().numpy()
            else:
                expert_action = distribution.sample().detach().cpu().numpy()

            assert expert_action.shape[0] == action_prob.shape[0] == 1
            action_prob = action_prob[0]
            expert_action = expert_action[0]
            if action_prob < 1 - self.config['free_level']:

                # print(f"Action probability: {action_prob}, agent action: {actions}, expert action: {expert_action},")

                if self.config["use_discrete"]:
                    expert_action = self.continuous_to_discrete(expert_action)
                    expert_action = self.discrete_to_continuous(expert_action)

                actions = expert_action

                self.takeover = True
            else:
                self.takeover = False
            # print(f"Action probability: {action_prob:.3f}, agent action: {actions}, expert action: {expert_action}, takeover: {self.takeover}")

        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config["disable_expert"]:
            i["takeover_log_prob"] = log_prob.item()










        # ----- Screenshot + VLM control (training-only) -----
        # Behavior:
        # 1) When not already waiting for the post-VLM period, manually capture a screenshot,
        #    then BLOCK the step until run_vlm() returns (this prevents the simulator from progressing
        #    while the VLM runs).
        # 2) After VLM returns, start a non-blocking timer (vlm_wait_time). While that timer
        #    is running the env behaves normally. When the timer expires we take the follow-up screenshot.
        if getattr(self, "_screenshotter", None) is not None:
            from pathlib import Path

            screenshot_dir = Path(self.config.get("screenshot_dir", "ima_log"))
            prefix = self.config.get("screenshot_prefix", "shot")
            vlm_wait = float(self.config.get("vlm_wait_time", self.config.get("screenshot_interval", 3.0)))

            # If we're in the post-VLM run period, check timer and capture when elapsed.
            if getattr(self, "_vlm_pending", False):
                if time.time() >= getattr(self, "_vlm_wait_until", 0.0):
                    # do the follow-up capture now and clear pending flag
                    try:
                        if hasattr(self._screenshotter, "capture"):
                            try:
                                self._screenshotter.capture(self.engine)
                            except TypeError:
                                self._screenshotter.capture()
                        else:
                            # fallback: force one maybe() call
                            self._screenshotter.maybe(self.engine)
                        print("[VLM] post-VLM follow-up screenshot captured.")
                    except Exception as e:
                        print(f"[Screenshotter] follow-up capture failed: {e}")
                    self._vlm_pending = False
                # else: still waiting; do nothing and allow simulator to continue running

            else:
                # Idle: trigger manual capture -> BLOCK until VLM responds -> schedule follow-up capture
                try:
                    # MANUAL initial capture (do not use periodic maybe())
                    if hasattr(self._screenshotter, "capture"):
                        try:
                            self._screenshotter.capture(self.engine)
                        except TypeError:
                            self._screenshotter.capture()
                    else:
                        # one-shot fallback
                        self._screenshotter.maybe(self.engine)
                except Exception as e:
                    print(f"[Screenshotter] initial capture failed: {e}")

                # find the latest screenshot file to send to VLM
                latest = None
                try:
                    if screenshot_dir.exists():
                        files = sorted(screenshot_dir.glob(f"{prefix}*"), key=lambda p: p.stat().st_mtime)
                    else:
                        files = []
                    if files:
                        latest = files[-1]
                    else:
                        print("[VLM] No screenshot files found to run VLM on.")
                except Exception as e:
                    print(f"[VLM integration] error while locating screenshot: {e}")

                # BLOCK the simulator here until the VLM returns by calling run_vlm synchronously.
                # This ensures the simulator does not progress during VLM inference.
                if latest is not None:
                    try:
                        vlm_result = run_vlm.run_vlm(str(latest))
                        print(f"[VLM] result for {latest.name}: {vlm_result}")
                    except Exception as e:
                        print(f"[VLM] error calling run_vlm on {latest}: {e}")
                else:
                    # No screenshot; skip VLM but still start the post-VLM wait to avoid spamming.
                    print("[VLM] skipped (no screenshot).")

                # Start the non-blocking post-VLM timer, let simulator continue running until it elapses.
                try:
                    self._vlm_pending = True
                    self._vlm_wait_until = time.time() + vlm_wait
                    # do NOT time.sleep() here — letting the environment step normally while we wait.
                    # The follow-up capture will occur in future step() calls when the timer elapses.
                except Exception as e:
                    print(f"[VLM integration] failed to start post-VLM timer: {e}")

































        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])

        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        """Compared to original one, here we don't call expert_policy, but directly get self.last_takeover."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


if __name__ == "__main__":
    env = FakeHumanEnv(dict(free_level=0.95, use_render=False))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        # done = tm or tc
        # env.render(mode="topdown")
        if done:
            print(info)
            env.reset()
