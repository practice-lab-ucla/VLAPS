import copy
import math
import pathlib
import time
import ipdb
import gymnasium as gym
import numpy as np
from metadrive.engine.logger import get_logger
from metadrive.policy.env_input_policy import EnvInputPolicy

from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from VLM import run_vlm

from .expert import LinearizedKinematicMPC, MPCConfig  # <-- MPC expert

FOLDER_PATH = pathlib.Path(__file__).parent
logger = get_logger()


class FakeHumanEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None

    def __init__(self, config):
        super(FakeHumanEnv, self).__init__(config)

        # --- MPC expert instance ---
        # One MPC per env; uses warm-start internally across steps.
        self.mpc = LinearizedKinematicMPC(MPCConfig())
        self.last_rrt_target_global = None

        # VLM-related state
        self._vlm_pending = False
        self._vlm_wait_until = 0.0

        # Discrete wrapper
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    # ------------------------------------------------------------------ #
    # Action space
    # ------------------------------------------------------------------ #
    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins**2)
        else:
            return super(FakeHumanEnv, self).action_space

    def default_config(self):
        """Use RL policy as 'agent', MPC (or VLM-assisted) as expert."""
        config = super(FakeHumanEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "free_level": 0.95,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": True,  # kept for compatibility

                # VLM/screenshot defaults
                "save_screenshots": False,
                "vlm_wait_time": 3.0,
                "screenshot_dir": "ima_log",
                "screenshot_prefix": "shot",
            },
            allow_add_new_key=True,
        )
        return config

    # ------------------------------------------------------------------ #
    # Discrete ↔ continuous helpers
    # ------------------------------------------------------------------ #
    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action

    # ------------------------------------------------------------------ #
    # Main step: AGENT + MPC expert + gate + VLM
    # ------------------------------------------------------------------ #



    def predict_bicycle_step(self, veh, action):
        """
        One-step kinematic bicycle prediction (no env stepping).
        action: [steer_norm, pedal_norm] in [-1, 1]^2
        Returns: (x_next, y_next), v_next, yaw_next in global frame
        """
        cfg = self.mpc.cfg
        dt = 3
        L = float(cfg.L)

        pos = np.array(veh.position, dtype=np.float64)
        heading_vec = np.array(veh.heading, dtype=np.float64)
        yaw = float(np.arctan2(heading_vec[1], heading_vec[0]))
        v = float(veh.speed)

        steer_norm, pedal_norm = float(action[0]), float(action[1])

        max_steer_rad = np.deg2rad(30.0)
        max_accel = 3.0

        delta = steer_norm * max_steer_rad
        a = pedal_norm * max_accel

        # Integrate speed and heading
        v_next = v + a * dt
        yaw_rate = v * np.tan(delta) / L
        yaw_next = yaw + yaw_rate * dt

        # Use the *updated* state (or a midpoint) to update position
        x_next = pos[0] + v_next * np.cos(yaw_next) * dt
        y_next = pos[1] + v_next * np.sin(yaw_next) * dt

        return np.array([x_next, y_next]), v_next, yaw_next
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def step(self, actions):
        """
        Same structure as your PPO/VLM env, but expert is now MPC:
        - Agent proposes actions (TD3 / RL).
        - MPC computes expert action from vehicle state.
        - Distance-based gate decides takeover.
        - VLM + screenshot logic kept as-is.
        """
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        # Store agent proposal
        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        takeover_log_prob = None  # PVP logger expects something here

        # ===== Expert takeover via MPC =====
        if not self.config["disable_expert"]:
            # Compute expert action from underlying MetaDrive vehicle state.
            # self.agent is the MetaDrive vehicle created by the base env.
            expert_action = self.mpc.action(self.agent, self.last_obs)  # [-1, 1]^2

            print(
            "AGENT (policy) action:", self.agent_action,
            " | EXPERT (MPC) action:", expert_action,
            )


            # Predict where we would go with EACH policy from the same current state:
            print("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
            agent_pos_next, agent_v_next, agent_yaw_next = self.predict_bicycle_step(
                self.agent, self.agent_action
            )
            expert_pos_next, expert_v_next, expert_yaw_next = self.predict_bicycle_step(
                self.agent, expert_action
            )


            print("AGENT predicted next global pos:", agent_pos_next)
            print("EXPERT predicted next global pos:", expert_pos_next)













            self.last_rrt_target_global = getattr(self.mpc, "_last_target_global", None)

            # Distance-based "probability-like" score (same as MPC env)
            diff = np.linalg.norm(actions - expert_action)
            max_diff = np.sqrt(8.0)  # max distance in [-1,1]^2
            diff_norm = np.clip(diff / max_diff, 0.0, 1.0)
            action_prob = 1.0 - diff_norm  # 1 if identical, ~0 if opposite

            # If actions disagree enough, let MPC take over
            # If you want the free_level semantics, uncomment the condition below
            # if action_prob < 1.0 - self.config["free_level"]:
            if True:
                if self.config["use_discrete"]:
                    expert_action_disc = self.continuous_to_discrete(expert_action)
                    expert_action = self.discrete_to_continuous(expert_action_disc)

                actions = expert_action
                self.takeover = True
            else:
                self.takeover = False

            takeover_log_prob = float(action_prob)  # log-prob substitute for logging

        # ===== Step the underlying HumanInTheLoopEnv =====
        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        # Extra logging expected by SharedControlMonitor / Monitor
        if not self.config["disable_expert"] and takeover_log_prob is not None:
            i["takeover_log_prob"] = takeover_log_prob

        # ------------------------------------------------------------------ #
        # Screenshot + VLM control (unchanged)
        # ------------------------------------------------------------------ #
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

                # BLOCK here until VLM returns
                if latest is not None:
                    try:
                        vlm_result = run_vlm.run_vlm(str(latest))
                        print(f"[VLM] result for {latest.name}: {vlm_result}")
                    except Exception as e:
                        print(f"[VLM] error calling run_vlm on {latest}: {e}")
                else:
                    print("[VLM] skipped (no screenshot).")

                # Start non-blocking post-VLM timer
                try:
                    self._vlm_pending = True
                    self._vlm_wait_until = time.time() + vlm_wait
                except Exception as e:
                    print(f"[VLM integration] failed to start post-VLM timer: {e}")

        # Sanity: takeover flag from here and from HumanInTheLoopEnv logic must match
        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])

        return o, r, d, i

    # ------------------------------------------------------------------ #
    # Takeover-cost bookkeeping (same pattern as before)
    # ------------------------------------------------------------------ #
    def _get_step_return(self, actions, engine_info):
        """Use self.last_takeover / self.takeover instead of policy-based flag."""
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
        _, _, done, info = env.step([0.0, 1.0])
        if done:
            print(info)
            env.reset()
