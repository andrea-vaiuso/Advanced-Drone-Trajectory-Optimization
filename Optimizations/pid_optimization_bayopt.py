# Author: Andrea Vaiuso
# Version: 2.2
# Date: 06.08.2025
# Description: Class-based Bayesian Optimization for PID gain tuning.
"""Bayesian Optimization for PID tuning packaged into a class."""

from time import time
from typing import Dict, Optional, List

import numpy as np
from bayes_opt import BayesianOptimization

import main as mainfunc
from Optimizations.opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)

from Optimizations.optimizer import Optimizer


class BayesianOptimizer(Optimizer):
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
        waypoints: Optional[List[dict]] = None,
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

        # Base trajectory and PID
        self.base_pid = mainfunc.load_pid_gains(self.parameters)
        params = self.parameters
        if waypoints is None:
            n_points = int(params.get("n_intermediate_waypoints", bayopt_cfg.get("n_points", 5)))
            A = np.array(params.get("start_point", bayopt_cfg.get("A", [0.0, 0.0, 0.0])), dtype=float)
            B = np.array(params.get("end_point", bayopt_cfg.get("B", [100.0, 100.0, 0.0])), dtype=float)
            line = np.linspace(A, B, n_points + 2)
            self.base_waypoints = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "v": 5}
                for p in line
            ]
        else:
            self.base_waypoints = waypoints
            n_points = len(self.base_waypoints) - 2
        self.n_points = n_points
        self.base_traj = np.array(
            [[wp["x"], wp["y"], wp["z"]] for wp in self.base_waypoints], dtype=float
        )

        pbounds_cfg = bayopt_cfg.get("pbounds", {})
        perturb = float(params.get("waypoint_perturbation_range", bayopt_cfg.get("perturbation_range", 300.0)))
        if pbounds_cfg:
            self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}
        else:
            self.pbounds = {
                f"p{i}_{ax}": (-perturb, perturb)
                for i in range(self.n_points)
                for ax in "xyz"
            }
        self.param_names = list(self.pbounds.keys())

        self.init_guess = {name: 0.0 for name in self.param_names}

        self.iteration = 0
        self.best_target = -np.inf
        self.best_params: Optional[List[dict]] = None
        self.costs: List[float] = []
        self.best_costs: List[float] = []

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_vector(self, vec: np.ndarray) -> List[dict]:
        """Convert a flat vector into a list of waypoints."""
        pts = self.base_traj.copy()
        vec = vec.reshape(self.n_points, 3)
        pts[1:-1] += vec
        return [
            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "v": 5}
            for p in pts
        ]

    def simulate_trajectory(self, waypoints: List[dict]) -> Dict[str, float]:
        """Run a simulation with the given trajectory and return the cost metrics."""
        return run_simulation(
            self.base_pid,
            self.parameters,
            waypoints,
            self.world,
            self.thrust_max,
            self.simulation_time,
            noise_model=self.noise_model,
            simulate_wind=self.simulate_wind_flag,
        )

    def _objective(self, **kwargs) -> float:
        """Objective function maximized by the Bayesian optimizer."""
        self.iteration += 1
        vec = np.array([kwargs[name] for name in self.param_names])
        waypoints = self.decode_vector(vec)
        sim_costs = self.simulate_trajectory(waypoints)
        total_cost = sim_costs["total_cost"]
        target = -total_cost

        log_step(
            {"waypoints": [(wp["x"], wp["y"], wp["z"]) for wp in waypoints]},
            total_cost,
            self.log_path,
            sim_costs,
        )
        if target > self.best_target:
            self.best_target = target
            self.best_params = waypoints
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
            vec = np.array([best[name] for name in self.param_names])
            best_waypoints = self.decode_vector(vec)
            global_best_cost = -optimizer.max["target"]
            show_best_params(
                "Bayesian",
                self.parameters,
                best_waypoints,
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
    optimizer = BayesianOptimizer(
        config_file="Settings/bay_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()

