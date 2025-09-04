# Author: Andrea Vaiuso
# Version: 1.3
# Date: 20.08.2025
# Description: Class-based implementation of a SAC-driven PID optimization
#              routine for a quadcopter controller. First attempt uses
#              initial PID gains when set_initial_obs=True. SAC random
#              pre-fill is disabled (learning_starts=0).
"""SAC optimization for PID tuning packaged into a class."""

from __future__ import annotations

from time import time
from typing import Dict, Optional

import main as mainfunc
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

from Optimizations.opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)

from Optimizations.optimizer import Optimizer


class SACPIDOptimizer(Optimizer):
    """Optimize PID gains using a Soft Actor-Critic agent.

    Parameters
    ----------
    config_file : str, optional
        Path to the SAC configuration file.
    parameters_file : str, optional
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Start each episode from the current PID gains when ``True`` and
        execute the first attempt with those gains.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is
        generated.
    """

    def __init__(
        self,
        config_file: str = "Settings/sac_opt.yaml",
        parameters_file: str = "Settings/simulation_parameters.yaml",
        *,
        verbose: bool = True,
        set_initial_obs: bool = True,
        simulate_wind_flag: bool = False,
        study_name: str = "",
        waypoints: Optional[list] = None,
        simulation_time: int = 150,
    ) -> None:
        super().__init__(
            "SAC",
            config_file,
            parameters_file,
            verbose=verbose,
            set_initial_obs=set_initial_obs,
            simulate_wind_flag=simulate_wind_flag,
            study_name=study_name,
            waypoints=waypoints,
            simulation_time=simulation_time,
        )

        sac_cfg = self.cfg

        self.total_timesteps = int(sac_cfg.get("total_timesteps", 1000))

        pbounds_cfg = sac_cfg.get("pbounds", {})
        self.pb_names = list(pbounds_cfg.keys())
        self.pb_low = np.array([pbounds_cfg[k][0] for k in self.pb_names], dtype=np.float32)
        self.pb_high = np.array([pbounds_cfg[k][1] for k in self.pb_names], dtype=np.float32)
        self.dim = len(self.pb_names)

        self.costs: list[float] = []
        self.best_costs: list[float] = []
        self.best_cost = np.inf
        self.best_params: Dict[str, tuple] | None = None
        self.step_count = 0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_action(self, vec: np.ndarray) -> Dict[str, tuple]:
        """Convert an action vector into a dictionary of PID gains."""
        return {
            "k_pid_pos": (vec[0], vec[1], vec[2]),
            "k_pid_alt": (vec[3], vec[4], vec[5]),
            "k_pid_att": (vec[6], vec[7], vec[8]),
            "k_pid_yaw": (0.5, 1e-6, 0.1),
            "k_pid_hsp": (vec[9], vec[10], vec[11]),
            "k_pid_vsp": (vec[12], vec[13], vec[14]),
        }

    def simulate_pid(self, pid_gains: Dict[str, tuple]) -> Dict[str, float]:
        """Run a simulation with the given PID gains and return the cost metrics."""
        return run_simulation(
            pid_gains,
            self.parameters,
            self.waypoints,
            self.world,
            self.thrust_max,
            self.simulation_time,
            noise_model=self.noise_model,
            simulate_wind=self.simulate_wind_flag,
        )

    # ------------------------------------------------------------------
    # Environment definition
    # ------------------------------------------------------------------
    class PIDEnv(gym.Env):
        """Gym environment in which each action represents a set of PID gains."""

        metadata = {"render.modes": []}

        def __init__(self, outer: "SACPIDOptimizer") -> None:
            super().__init__()
            self.outer = outer
            self.action_space = spaces.Box(
                outer.pb_low, outer.pb_high, dtype=np.float32
            )
            self.observation_space = spaces.Box(
                outer.pb_low, outer.pb_high, dtype=np.float32
            )
            current_best = mainfunc.load_pid_gains(outer.parameters)
            self.initial_obs = np.array(
                [
                    current_best["k_pid_pos"][0],
                    current_best["k_pid_pos"][1],
                    current_best["k_pid_pos"][2],
                    current_best["k_pid_alt"][0],
                    current_best["k_pid_alt"][1],
                    current_best["k_pid_alt"][2],
                    current_best["k_pid_att"][0],
                    current_best["k_pid_att"][1],
                    current_best["k_pid_att"][2],
                    current_best["k_pid_hsp"][0],
                    current_best["k_pid_hsp"][1],
                    current_best["k_pid_hsp"][2],
                    current_best["k_pid_vsp"][0],
                    current_best["k_pid_vsp"][1],
                    current_best["k_pid_vsp"][2],
                ],
                dtype=np.float32,
            )
            if outer.set_initial_obs:
                self.state = self.initial_obs.copy()
            else:
                self.state = np.zeros(outer.dim, dtype=np.float32)

        def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
            super().reset(seed=seed)
            if self.outer.set_initial_obs:
                self.state = self.initial_obs.copy()
            else:
                self.state = np.zeros(self.outer.dim, dtype=np.float32)
            return self.state, {}

        def step(self, action: np.ndarray):  # type: ignore[override]
            """Take a step in the environment using the provided action."""
            self.outer.step_count += 1

            # If initial observations are requested, execute the very first attempt
            # with those gains (clipped to bounds). No extra flags: relies on step_count.
            if self.outer.set_initial_obs and self.outer.step_count == 1:
                action_to_use = np.clip(self.initial_obs, self.outer.pb_low, self.outer.pb_high)
            else:
                action_to_use = np.clip(action, self.outer.pb_low, self.outer.pb_high)

            gains = self.outer.decode_action(action_to_use)
            sim_costs = self.outer.simulate_pid(gains)
            total_cost = sim_costs["total_cost"]
            reward = -total_cost

            log_step(gains, total_cost, self.outer.log_path, sim_costs)
            self.outer.costs.append(total_cost)
            if total_cost < self.outer.best_cost:
                self.outer.best_cost = total_cost
                self.outer.best_params = gains
            self.outer.best_costs.append(self.outer.best_cost)

            self.state = action_to_use.astype(np.float32)
            terminated = True  # one-step episode
            truncated = False
            info = {"costs": sim_costs}
            if self.outer.verbose:
                first = " (first attempt with initial gains)" if (self.outer.set_initial_obs and self.outer.step_count == 1) else ""
                print(
                    f"[ SAC ] Step ({self.outer.step_count}/{self.outer.total_timesteps}) "
                    f"completed: total_cost={total_cost:.4f}, best_cost={self.outer.best_cost:.4f}, costs: {sim_costs}{first}"
                )
            return self.state, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Optimization routine
    # ------------------------------------------------------------------
    def optimize(self) -> None:
        """Execute the SAC optimization process."""
        env = self.PIDEnv(self)

        # Disable random pre-fill
        model = SAC("MlpPolicy", env, learning_starts=0, verbose=int(self.verbose))

        start_opt = time()
        print("Starting SAC optimization...")
        try:
            model.learn(total_timesteps=self.total_timesteps, progress_bar=False)
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        finally:
            tot_time = time() - start_opt
            if self.best_params is None:
                print("No evaluations were performed.")
                return
            show_best_params(
                "RL - SAC",
                self.parameters,
                self.best_params,
                self.opt_output_path,
                self.best_cost,
                self.step_count,
                self.simulation_time,
                tot_time,
            )
            plot_costs_trend(
                self.costs,
                save_path=self.opt_output_path.replace(".txt", "_costs.png"),
                alg_name="RL - SAC",
            )
            plot_costs_trend(
                self.best_costs,
                save_path=self.opt_output_path.replace(".txt", "_best_costs.png"),
                alg_name="RL - SAC",
            )


def main() -> None:
    """Run PID optimization using SAC."""
    optimizer = SACPIDOptimizer(
        config_file="Settings/sac_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
        waypoints=mainfunc.create_training_waypoints(),
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()
