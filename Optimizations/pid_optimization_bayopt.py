# Author: Andrea Vaiuso
# Version: 2.2
# Date: 06.08.2025
# Description: Class-based Bayesian Optimization for PID gain tuning.
"""Bayesian Optimization for PID tuning packaged into a class."""

from time import time
from typing import Dict, Optional

import numpy as np
from bayes_opt import BayesianOptimization

import main as mainfunc
from opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)

from optimizer import Optimizer


class BayesianPIDOptimizer(Optimizer):
    """Optimize PID gains using Bayesian Optimization.

    Parameters
    ----------
    config_file : str, optional
        Path to the Bayesian optimization configuration file.
    parameters_file : str, optional
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Probe the current PID gains before the optimization when ``True``.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is
        generated.
    """

    def __init__(
        self,
        config_file: str = "Settings/bay_opt.yaml",
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
            "Bayesian",
            config_file,
            parameters_file,
            verbose=verbose,
            set_initial_obs=set_initial_obs,
            simulate_wind_flag=simulate_wind_flag,
            study_name=study_name,
            waypoints=waypoints,
            simulation_time=simulation_time,
        )

        bayopt_cfg = self.cfg

        self.n_iter = int(bayopt_cfg.get("n_iter", 1500))
        self.init_points = int(bayopt_cfg.get("init_points", 20))

        pbounds_cfg = bayopt_cfg.get("pbounds", {})
        self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}

        current_best = mainfunc.load_pid_gains(self.parameters)
        self.init_guess = {
            "kp_pos": current_best["k_pid_pos"][0],
            "ki_pos": current_best["k_pid_pos"][1],
            "kd_pos": current_best["k_pid_pos"][2],
            "kp_alt": current_best["k_pid_alt"][0],
            "ki_alt": current_best["k_pid_alt"][1],
            "kd_alt": current_best["k_pid_alt"][2],
            "kp_att": current_best["k_pid_att"][0],
            "ki_att": current_best["k_pid_att"][1],
            "kd_att": current_best["k_pid_att"][2],
            "kp_hsp": current_best["k_pid_hsp"][0],
            "ki_hsp": current_best["k_pid_hsp"][1],
            "kd_hsp": current_best["k_pid_hsp"][2],
            "kp_vsp": current_best["k_pid_vsp"][0],
            "ki_vsp": current_best["k_pid_vsp"][1],
            "kd_vsp": current_best["k_pid_vsp"][2],
        }

        self.iteration = 0
        self.best_target = -np.inf
        self.best_params: Optional[Dict[str, tuple]] = None
        self.costs: list[float] = []
        self.best_costs: list[float] = []

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
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

    def _objective(self, **kwargs) -> float:
        """Objective function maximized by the Bayesian optimizer."""
        self.iteration += 1
        params = {
            "k_pid_pos": (kwargs["kp_pos"], kwargs["ki_pos"], kwargs["kd_pos"]),
            "k_pid_alt": (kwargs["kp_alt"], kwargs["ki_alt"], kwargs["kd_alt"]),
            "k_pid_att": (kwargs["kp_att"], kwargs["ki_att"], kwargs["kd_att"]),
            "k_pid_yaw": (0.5, 1e-6, 0.1),
            "k_pid_hsp": (kwargs["kp_hsp"], kwargs["ki_hsp"], kwargs["kd_hsp"]),
            "k_pid_vsp": (kwargs["kp_vsp"], kwargs["ki_vsp"], kwargs["kd_vsp"]),
        }
        sim_costs = self.simulate_pid(params)
        total_cost = sim_costs["total_cost"]
        target = -total_cost

        log_step(params, total_cost, self.log_path, sim_costs)
        if target > self.best_target:
            self.best_target = target
            self.best_params = params
        self.costs.append(total_cost)
        self.best_costs.append(-self.best_target)

        if self.verbose:
            print(
                f"[ BAY_OPT ] {self.iteration}/{self.n_iter}: cost={total_cost:.4f}, "
                f"best_cost={-self.best_target:.4f}, costs={sim_costs}"
            )
        return target

    # ------------------------------------------------------------------
    # Optimization routine
    # ------------------------------------------------------------------
    def optimize(self) -> None:
        """Execute the Bayesian Optimization process."""
        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.pbounds,
            random_state=42,
        )
        if self.set_initial_obs:
            optimizer.probe(params=self.init_guess, lazy=True)

        start_time = time()
        print("Starting Bayesian Optimization...")
        try:
            optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        finally:
            tot_time = time() - start_time
            if not optimizer.res:
                print("No evaluations were performed.")
                return
            best = optimizer.max["params"]
            best_formatted = {
                "k_pid_pos": (best["kp_pos"], best["ki_pos"], best["kd_pos"]),
                "k_pid_alt": (best["kp_alt"], best["ki_alt"], best["kd_alt"]),
                "k_pid_att": (best["kp_att"], best["ki_att"], best["kd_att"]),
                "k_pid_yaw": (0.5, 1e-6, 0.1),
                "k_pid_hsp": (best["kp_hsp"], best["ki_hsp"], best["kd_hsp"]),
                "k_pid_vsp": (best["kp_vsp"], best["ki_vsp"], best["kd_vsp"]),
            }
            global_best_cost = -optimizer.max["target"]
            show_best_params(
                "Bayesian",
                self.parameters,
                best_formatted,
                self.opt_output_path,
                global_best_cost,
                self.iteration,
                self.simulation_time,
                tot_time,
            )
            plot_costs_trend(
                self.costs,
                save_path=self.opt_output_path.replace(".txt", "_costs.png"),
                alg_name="Bayesian Optimization",
            )
            plot_costs_trend(
                self.best_costs,
                save_path=self.opt_output_path.replace(".txt", "_best_costs.png"),
                alg_name="Bayesian Optimization",
            )


def main() -> None:
    """Run PID optimization using Bayesian Optimization."""
    optimizer = BayesianPIDOptimizer(
        config_file="Settings/bay_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
        waypoints=mainfunc.create_training_waypoints(),
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()

