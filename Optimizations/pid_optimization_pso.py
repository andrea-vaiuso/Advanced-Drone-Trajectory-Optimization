# Author: Andrea Vaiuso
# Version: 1.3
# Date: 06.08.2025
# Description: Class-based Particle Swarm Optimization for PID gain tuning.
"""Particle Swarm Optimization for PID tuning packaged into a class."""

from time import time
from typing import Dict, Optional, List

import numpy as np

import main as mainfunc
from Optimizations.opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)


from Optimizations.optimizer import Optimizer


class PSOOptimizer(Optimizer):
    """Optimize PID gains using Particle Swarm Optimization.

    Parameters
    ----------
    config_file : str, optional
        Path to the PSO configuration file.
    parameters_file : str, optional
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Include current PID gains as the first particle when ``True``.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is
        generated.
    """

    def __init__(
        self,
        config_file: str = "Settings/pso_opt.yaml",
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
            "PSO",
            config_file,
            parameters_file,
            verbose=verbose,
            set_initial_obs=set_initial_obs,
            simulate_wind_flag=simulate_wind_flag,
            study_name=study_name,
            waypoints=waypoints,
            simulation_time=simulation_time,
        )

        pso_cfg = self.cfg

        self.n_iter = int(pso_cfg.get("n_iter", 100))
        self.swarm_size = int(pso_cfg.get("swarm_size", 30))
        self.w = float(pso_cfg.get("inertia_weight", 0.7))
        self.c1 = float(pso_cfg.get("cognitive_coeff", 1.5))
        self.c2 = float(pso_cfg.get("social_coeff", 1.5))

        # Base trajectory and PID
        self.base_pid = mainfunc.load_pid_gains(self.parameters)
        params = self.parameters
        if waypoints is None:
            n_points = int(params.get("n_intermediate_waypoints", pso_cfg.get("n_points", 5)))
            A = np.array(params.get("start_point", pso_cfg.get("A", [0.0, 0.0, 0.0])), dtype=float)
            B = np.array(params.get("end_point", pso_cfg.get("B", [100.0, 100.0, 0.0])), dtype=float)
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

        pbounds_cfg = pso_cfg.get("pbounds", {})
        perturb = float(params.get("waypoint_perturbation_range", pso_cfg.get("perturbation_range", 300.0)))
        if pbounds_cfg:
            self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}
            self.lower_bounds = np.array(
                [v[0] for v in self.pbounds.values()], dtype=float
            )
            self.upper_bounds = np.array(
                [v[1] for v in self.pbounds.values()], dtype=float
            )
        else:
            self.lower_bounds = -perturb * np.ones(self.n_points * 3)
            self.upper_bounds = perturb * np.ones(self.n_points * 3)
            self.pbounds = {
                f"p{i}_{ax}": (self.lower_bounds[3 * i + j], self.upper_bounds[3 * i + j])
                for i in range(self.n_points)
                for j, ax in enumerate("xyz")
            }
        self.dim = self.lower_bounds.size

        self.costs: List[float] = []
        self.best_costs: List[float] = []
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_cost = np.inf

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_particle(self, vec: np.ndarray) -> List[dict]:
        """Convert a particle vector into a list of waypoints."""
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

    # ------------------------------------------------------------------
    # Optimization routine
    # ------------------------------------------------------------------
    def optimize(self) -> None:
        """Execute the Particle Swarm Optimization process."""
        rng = np.random.default_rng(42)
        particles_pos = rng.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.swarm_size, self.dim)
        )
        particles_vel = np.zeros((self.swarm_size, self.dim))

        if self.set_initial_obs:
            particles_pos[0] = np.zeros(self.dim)

        personal_best_pos = particles_pos.copy()
        personal_best_cost = np.full(self.swarm_size, np.inf)

        start_opt = time()
        print("Starting Particle Swarm Optimization...")
        try:
            for gen in range(self.n_iter):
                for i in range(self.swarm_size):
                    self.iteration = (i + 1) * (gen + 1)
                    waypoints = self.decode_particle(particles_pos[i])
                    costs_sim = self.simulate_trajectory(waypoints)
                    total_cost = costs_sim["total_cost"]
                    self.costs.append(total_cost)
                    log_step(
                        {"waypoints": [(wp["x"], wp["y"], wp["z"]) for wp in waypoints]},
                        total_cost,
                        self.log_path,
                        costs_sim,
                    )
                    if self.verbose:
                        print(
                            f"[ PSO ] Iteration {self.iteration}/{self.n_iter*self.swarm_size} |"
                            f"cost={total_cost:.4f}, costs={costs_sim}"
                        )
                    if total_cost < personal_best_cost[i]:
                        personal_best_cost[i] = total_cost
                        personal_best_pos[i] = particles_pos[i].copy()
                    if total_cost < self.global_best_cost:
                        self.global_best_cost = total_cost
                        self.global_best_pos = particles_pos[i].copy()
                    self.best_costs.append(self.global_best_cost)
                for i in range(self.swarm_size):
                    r1 = rng.random(self.dim)
                    r2 = rng.random(self.dim)
                    particles_vel[i] = (
                        self.w * particles_vel[i]
                        + self.c1 * r1 * (personal_best_pos[i] - particles_pos[i])
                        + self.c2 * r2 * (self.global_best_pos - particles_pos[i])
                    )
                    particles_pos[i] = particles_pos[i] + particles_vel[i]
                    particles_pos[i] = np.clip(
                        particles_pos[i], self.lower_bounds, self.upper_bounds
                    )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        finally:
            tot_time = time() - start_opt
            if self.global_best_pos is None:
                print("No evaluations were performed.")
                return
            best_params = self.decode_particle(self.global_best_pos)
            show_best_params(
                "Particle Swarm Optimization",
                self.parameters,
                best_params,
                self.opt_output_path,
                self.global_best_cost,
                self.iteration,
                self.simulation_time,
                tot_time,
            )
            plot_costs_trend(
                self.costs,
                save_path=self.opt_output_path.replace(".txt", "_costs.png"),
                alg_name="Particle Swarm Optimization",
            )
            plot_costs_trend(
                self.best_costs,
                save_path=self.opt_output_path.replace(".txt", "_best_costs.png"),
                alg_name="Particle Swarm Optimization",
            )


def main() -> None:
    """Run PID optimization using Particle Swarm Optimization."""
    optimizer = PSOOptimizer(
        config_file="Settings/pso_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()

