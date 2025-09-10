# Author: Andrea Vaiuso
# Version: 1.0
# Date: 06.08.2025
# Description: Class-based Genetic Algorithm for PID gain tuning.
"""Genetic Algorithm for PID tuning packaged into a class."""

from time import time
from typing import Dict, Optional, Tuple

import numpy as np

import main as mainfunc
from Optimizations.opt_func import (
    log_step,
    plot_costs_trend,
    show_best_params,
    run_simulation,
)

from Optimizations.optimizer import Optimizer


class GAOptimizer(Optimizer):
    """Optimize PID gains using a Genetic Algorithm.

    Parameters
    ----------
    config_file : str, optional
        Path to the GA configuration file.
    parameters_file : str, optional
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Include current PID gains as the first individual when ``True``.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is
        generated.
    """

    def __init__(
        self,
        config_file: str = "Settings/ga_opt.yaml",
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
            "GA",
            config_file,
            parameters_file,
            verbose=verbose,
            set_initial_obs=set_initial_obs,
            simulate_wind_flag=simulate_wind_flag,
            study_name=study_name,
            waypoints=waypoints,
            simulation_time=simulation_time,
        )

        ga_cfg = self.cfg

        self.n_generations = int(ga_cfg.get("n_generations", 100))
        self.population_size = int(ga_cfg.get("population_size", 30))
        self.crossover_rate = float(ga_cfg.get("crossover_rate", 0.8))
        self.mutation_rate = float(ga_cfg.get("mutation_rate", 0.1))
        self.tournament_size = int(ga_cfg.get("tournament_size", 3))
        self.elite_fraction = float(ga_cfg.get("elite_fraction", 0.1))

        # Set up base trajectory between start and end points
        self.base_pid = mainfunc.load_pid_gains(self.parameters)
        if waypoints is None:
            n_points = int(ga_cfg.get("n_points", 5))
            A = np.array(ga_cfg.get("A", [0.0, 0.0, 0.0]), dtype=float)
            B = np.array(ga_cfg.get("B", [100.0, 100.0, 0.0]), dtype=float)
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

        pbounds_cfg = ga_cfg.get("pbounds", {})
        if pbounds_cfg:
            self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}
            self.lower_bounds = np.array(
                [v[0] for v in self.pbounds.values()], dtype=float
            )
            self.upper_bounds = np.array(
                [v[1] for v in self.pbounds.values()], dtype=float
            )
        else:
            perturb = float(ga_cfg.get("perturbation_range", 10.0))
            self.lower_bounds = -perturb * np.ones(self.n_points * 3)
            self.upper_bounds = perturb * np.ones(self.n_points * 3)
            self.pbounds = {
                f"p{i}_{ax}": (self.lower_bounds[3 * i + j], self.upper_bounds[3 * i + j])
                for i in range(self.n_points)
                for j, ax in enumerate("xyz")
            }
        self.dim = self.lower_bounds.size

        self.costs: list[float] = []
        self.best_costs: list[float] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_cost = np.inf
        self.elite_count = max(1, int(self.elite_fraction * self.population_size))

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_individual(self, vec: np.ndarray) -> list[dict]:
        """Convert an individual vector into a list of waypoints."""
        pts = self.base_traj.copy()
        vec = vec.reshape(self.n_points, 3)
        pts[1:-1] += vec
        return [
            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "v": 5}
            for p in pts
        ]

    def simulate_trajectory(self, waypoints: list) -> Dict[str, float]:
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

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Select a parent using tournament selection."""
        idxs = rng.integers(0, self.population_size, size=self.tournament_size)
        best_idx = idxs[np.argmin(fitness[idxs])]
        return population[best_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Perform single-point crossover between two parents."""
        if rng.random() < self.crossover_rate:
            point = rng.integers(1, self.dim)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutate an individual by random resetting within bounds."""
        for j in range(self.dim):
            if rng.random() < self.mutation_rate:
                individual[j] = rng.uniform(self.lower_bounds[j], self.upper_bounds[j])
        return np.clip(individual, self.lower_bounds, self.upper_bounds)

    # ------------------------------------------------------------------
    # Optimization routine
    # ------------------------------------------------------------------
    def optimize(self) -> None:
        """Execute the Genetic Algorithm optimization process."""
        rng = np.random.default_rng(42)
        population = rng.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.population_size, self.dim)
        )

        if self.set_initial_obs:
            population[0] = np.zeros(self.dim)

        fitness = np.full(self.population_size, np.inf)
        start_opt = time()
        print("Starting Genetic Algorithm optimization...")
        try:
            for gen in range(self.n_generations):
                # Evaluate population
                for i in range(self.population_size):
                    self.iteration = (i + 1) * (gen + 1)
                    waypoints = self.decode_individual(population[i])
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
                            f"[ GA ] Iteration {self.iteration}/{self.n_generations*self.population_size} | "
                            f"cost={total_cost:.4f}, costs={costs_sim}"
                        )
                    fitness[i] = total_cost
                    if total_cost < self.best_cost:
                        self.best_cost = total_cost
                        self.best_individual = population[i].copy()
                    self.best_costs.append(self.best_cost)

                # Build next generation
                new_population = []
                elite_indices = np.argsort(fitness)[: self.elite_count]
                new_population.extend(population[elite_indices])
                while len(new_population) < self.population_size:
                    p1 = self._tournament_selection(population, fitness, rng)
                    p2 = self._tournament_selection(population, fitness, rng)
                    c1, c2 = self._crossover(p1, p2, rng)
                    c1 = self._mutate(c1, rng)
                    new_population.append(c1)
                    if len(new_population) < self.population_size:
                        c2 = self._mutate(c2, rng)
                        new_population.append(c2)
                population = np.array(new_population[: self.population_size])
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        finally:
            tot_time = time() - start_opt
            if self.best_individual is None:
                print("No evaluations were performed.")
                return
            best_params = self.decode_individual(self.best_individual)
            show_best_params(
                "Genetic Algorithm",
                self.parameters,
                best_params,
                self.opt_output_path,
                self.best_cost,
                self.iteration,
                self.simulation_time,
                tot_time,
            )
            plot_costs_trend(
                self.costs,
                save_path=self.opt_output_path.replace(".txt", "_costs.png"),
                alg_name="Genetic Algorithm",
            )
            plot_costs_trend(
                self.best_costs,
                save_path=self.opt_output_path.replace(".txt", "_best_costs.png"),
                alg_name="Genetic Algorithm",
            )


def main() -> None:
    """Run PID optimization using a Genetic Algorithm."""
    optimizer = GAOptimizer(
        config_file="Settings/ga_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()
