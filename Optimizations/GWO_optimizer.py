"""Grey Wolf Optimizer implementation for waypoint perturbations."""

import numpy as np

from Drone.Simulation import Simulation
from Optimizations.optimizer import Optimizer


class GWOOptimizer(Optimizer):
    """Optimize waypoint perturbations with the Grey Wolf metaheuristic.

    Each wolf in the pack represents a candidate perturbation of the neutral
    A→B segment.  The algorithm promotes the three best wolves (alpha, beta,
    delta) and lets the remaining pack members update their offsets by
    averaging the leaders.  The helper methods inherited from
    :class:`MetaHeuristicOptimizer` take care of running the simulator and
    logging the results so this class focuses purely on the search logic.
    """

    def __init__(
        self,
        simulation_object: Simulation,
        A,
        B,
        config_file: str = "Settings/metaheuristic_parameters.yaml",
        verbose: bool = True,
    ) -> None:
        """Read the Grey Wolf configuration and prepare the optimiser.

        Args:
            simulation_object: Shared simulator instance.
            A: Start point of the mission (x, y, z).
            B: Target point of the mission (x, y, z).
            config_file: Path to :mod:`Settings/metaheuristic_parameters.yaml`.
            verbose: When ``True`` prints a progress line after every
                generation.
        """
        super().__init__(
            simulation_object=simulation_object,
            opt_method_name="GWO",
            config_file=config_file,
            verbose=verbose,
        )
        self.A = tuple(float(v) for v in A)
        self.B = tuple(float(v) for v in B)

        shared_cfg = dict(self.cfg.get("shared", {}))
        algo_cfg = dict(self.cfg.get("gwo", self.cfg))
        self.cfg = {"shared": shared_cfg, "gwo": algo_cfg}

        self.n_points = int(algo_cfg.get("n_points", shared_cfg.get("n_points", 5)))
        self.n_generations = int(algo_cfg.get("n_generations", 100))
        self.pack_size = int(algo_cfg.get("pack_size", 30))
        self.max_perturbation_offset = float(
            algo_cfg.get(
                "max_perturbation_offset",
                shared_cfg.get("max_perturbation_offset", 250.0),
            )
        )
        self.max_velocity = float(
            algo_cfg.get("max_velocity", shared_cfg.get("max_velocity", 20.0))
        )
        self.random_seed = int(
            algo_cfg.get("random_seed", shared_cfg.get("random_seed", 42))
        )

        self.global_best_cost = float("inf")
        self.global_best_pos = None

    def optimize(self, seed=None):
        """Run the Grey Wolf search and return the best waypoint set found.

        Args:
            seed: Optional seed overriding the configuration file.  Using the
                same seed yields reproducible populations and trajectories.

        Returns:
            List of waypoint dictionaries describing the best trajectory.
            Returns an empty list when no feasible solution is identified.
        """

        world_max = getattr(self.simulation_object.world, "max_world_size", 1000.0)
        low_bounds, high_bounds = self.build_particle_bounds(
            self.A,
            self.B,
            self.n_points,
            self.max_perturbation_offset,
            self.max_velocity,
            world_min=0.0,
            world_max=float(world_max),
        )

        rng = np.random.default_rng(self.random_seed if seed is None else seed)
        wolves_pos = rng.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(self.pack_size, self.n_points * 4),
        )
        if self.set_initial_obs:
            neutral_speed = 5.0
            wolves_pos[0] = np.tile([0.0, 0.0, 0.0, neutral_speed], self.n_points)

        personal_best_pos = wolves_pos.copy()
        personal_best_cost = np.full(self.pack_size, np.inf)

        self.global_best_cost = float("inf")
        self.global_best_pos = None

        try:
            for gen in range(self.n_generations):
                for idx in range(self.pack_size):
                    waypoint_offsets = wolves_pos[idx]
                    waypoints = self.decode_particle(
                        waypoint_offsets, self.n_points, self.A, self.B
                    )
                    costs_sim = self.run_simulation(waypoints)
                    total_cost = float(costs_sim["total_cost"])
                    self.log_step(waypoints)

                    if total_cost < personal_best_cost[idx]:
                        personal_best_cost[idx] = total_cost
                        personal_best_pos[idx] = waypoint_offsets.copy()
                    if total_cost < self.global_best_cost:
                        self.global_best_cost = total_cost
                        self.global_best_pos = waypoint_offsets.copy()

                a = 2 - gen * (2 / max(self.n_generations, 1))
                sorted_indices = np.argsort(personal_best_cost)
                alpha_pos = personal_best_pos[sorted_indices[0]]
                beta_pos = personal_best_pos[sorted_indices[1]]
                delta_pos = personal_best_pos[sorted_indices[2]]

                for idx in range(self.pack_size):
                    for dim in range(self.n_points * 4):
                        r1, r2 = rng.random(), rng.random()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[dim] - wolves_pos[idx][dim])
                        X1 = alpha_pos[dim] - A1 * D_alpha

                        r1, r2 = rng.random(), rng.random()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[dim] - wolves_pos[idx][dim])
                        X2 = beta_pos[dim] - A2 * D_beta

                        r1, r2 = rng.random(), rng.random()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[dim] - wolves_pos[idx][dim])
                        X3 = delta_pos[dim] - A3 * D_delta

                        wolves_pos[idx][dim] = (X1 + X2 + X3) / 3

                    wolves_pos[idx] = np.clip(
                        wolves_pos[idx],
                        low_bounds,
                        high_bounds,
                    )

                if self.verbose:
                    running_best = (
                        min(self.costs_history) if self.costs_history else self.global_best_cost
                    )
                    print(
                        f"{self.get_alg_prefix()} Gen {gen + 1}/{self.n_generations}, "
                        f"Best Cost: {self.global_best_cost:.2f}, "
                        f"History Best: {running_best:.2f}"
                    )
        except KeyboardInterrupt:
            print(f"{self.get_alg_prefix()} Optimization interrupted by user.")
        finally:
            if self.global_best_pos is None:
                print(f"{self.get_alg_prefix()} No valid solution found.")
                return []

            return self.decode_particle(
                self.global_best_pos, self.n_points, self.A, self.B
            )
