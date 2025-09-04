# Author: Andrea Vaiuso
# Version: 1.3
# Date: 06.08.2025
# Description: Class-based Particle Swarm Optimization for PID gain tuning.
"""Particle Swarm Optimization for PID tuning packaged into a class."""

from time import time
from typing import Dict, Optional

import numpy as np

import main as mainfunc
from opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)


from optimizer import Optimizer


class PSOPIDOptimizer(Optimizer):
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
        waypoints: Optional[list] = None,
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

        pbounds_cfg = pso_cfg.get("pbounds", {})
        self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}
        self.lower_bounds = np.array([v[0] for v in self.pbounds.values()], dtype=float)
        self.upper_bounds = np.array([v[1] for v in self.pbounds.values()], dtype=float)
        self.dim = len(self.lower_bounds)

        self.costs: list[float] = []
        self.best_costs: list[float] = []
        self.global_best_pos: Optional[np.ndarray] = None
        self.global_best_cost = np.inf

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_particle(self, vec: np.ndarray) -> Dict[str, tuple]:
        """Convert a particle vector into a PID gain dictionary."""
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
            current_best = mainfunc.load_pid_gains(self.parameters)
            init_particle = np.array(
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
                ]
            )
            particles_pos[0] = np.clip(init_particle, self.lower_bounds, self.upper_bounds)

        personal_best_pos = particles_pos.copy()
        personal_best_cost = np.full(self.swarm_size, np.inf)

        start_opt = time()
        print("Starting Particle Swarm Optimization...")
        try:
            for gen in range(self.n_iter):
                for i in range(self.swarm_size):
                    self.iteration = (i + 1) * (gen + 1)
                    gains = self.decode_particle(particles_pos[i])
                    costs_sim = self.simulate_pid(gains)
                    total_cost = costs_sim["total_cost"]
                    self.costs.append(total_cost)
                    log_step(gains, total_cost, self.log_path, costs_sim)
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
    optimizer = PSOPIDOptimizer(
        config_file="Settings/pso_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
        waypoints=mainfunc.create_training_waypoints(),
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()

