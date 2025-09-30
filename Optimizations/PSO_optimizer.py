"""Particle Swarm Optimizer implementation for waypoint perturbations."""

import numpy as np

from Drone.Simulation import Simulation
from Optimizations.optimizer import MetaHeuristicOptimizer


class PSOOptimizer(MetaHeuristicOptimizer):
    """Search waypoint perturbations using Particle Swarm Optimization.

    Every particle stores a vector of perturbations relative to the neutral
    straight line from start to target.  The swarm iteratively combines its
    personal best experiences with the global best individual to explore the
    search space.  Logging and cost computations are delegated to
    :class:`MetaHeuristicOptimizer`.
    """

    def __init__(
        self,
        simulation_object: Simulation,
        A,
        B,
        config_file: str = "Settings/metaheuristic_parameters.yaml",
        verbose: bool = True,
    ) -> None:
        """Read the PSO configuration and set up the swarm.

        Args:
            simulation_object: Shared simulator instance.
            A: Start point of the mission (x, y, z).
            B: Target point of the mission (x, y, z).
            config_file: Path to :mod:`Settings/metaheuristic_parameters.yaml`.
            verbose: When ``True`` prints a progress line after each generation.
        """
        super().__init__(
            simulation_object=simulation_object,
            opt_method_name="PSO",
            config_file=config_file,
            verbose=verbose,
        )
        self.A = tuple(float(v) for v in A)
        self.B = tuple(float(v) for v in B)

        shared_cfg = dict(self.cfg.get("shared", {}))
        algo_cfg = dict(self.cfg.get("pso", self.cfg))
        self.cfg = {"shared": shared_cfg, "pso": algo_cfg}

        self.n_points = int(algo_cfg.get("n_points", shared_cfg.get("n_points", 5)))
        self.n_generations = int(algo_cfg.get("n_generations", 100))
        self.swarm_size = int(algo_cfg.get("swarm_size", 30))
        self.max_perturbation_offset = float(
            algo_cfg.get(
                "max_perturbation_offset",
                shared_cfg.get("max_perturbation_offset", 250.0),
            )
        )
        self.max_velocity = float(
            algo_cfg.get("max_velocity", shared_cfg.get("max_velocity", 20.0))
        )
        self.inertia = float(algo_cfg.get("inertia", 0.5))
        self.cognitive_coeff = float(algo_cfg.get("cognitive_coeff", 1.5))
        self.social_coeff = float(algo_cfg.get("social_coeff", 1.5))
        self.random_seed = int(
            algo_cfg.get("random_seed", shared_cfg.get("random_seed", 42))
        )

        self.global_best_cost = float("inf")
        self.global_best_pos = None

    def optimize(self, seed=None):
        """Run the PSO search and return the best waypoint set found.

        Args:
            seed: Optional seed overriding the configuration file to ensure
                reproducible swarms.

        Returns:
            List of waypoint dictionaries describing the best trajectory, or an
            empty list if a feasible solution cannot be located.
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
        particles_pos = rng.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(self.swarm_size, self.n_points * 4),
        )
        particles_vel = np.zeros_like(particles_pos)
        if self.set_initial_obs:
            neutral_speed = 5.0
            particles_pos[0] = np.tile([0.0, 0.0, 0.0, neutral_speed], self.n_points)

        personal_best_pos = particles_pos.copy()
        personal_best_cost = np.full(self.swarm_size, np.inf)

        self.global_best_cost = float("inf")
        self.global_best_pos = None

        try:
            for gen in range(self.n_generations):
                for idx in range(self.swarm_size):
                    offsets = particles_pos[idx]
                    waypoints = self.decode_particle(offsets, self.n_points, self.A, self.B)
                    costs_sim = self.run_simulation(waypoints)
                    total_cost = float(costs_sim["total_cost"])
                    self.log_step(waypoints)

                    if total_cost < personal_best_cost[idx]:
                        personal_best_cost[idx] = total_cost
                        personal_best_pos[idx] = offsets.copy()
                    if total_cost < self.global_best_cost:
                        self.global_best_cost = total_cost
                        self.global_best_pos = offsets.copy()

                for idx in range(self.swarm_size):
                    r1, r2 = rng.random(2)
                    # The global best is guaranteed to be available after the evaluation loop above.
                    assert self.global_best_pos is not None
                    cognitive_vel = self.cognitive_coeff * r1 * (
                        personal_best_pos[idx] - particles_pos[idx]
                    )
                    social_vel = self.social_coeff * r2 * (
                        self.global_best_pos - particles_pos[idx]
                    )
                    particles_vel[idx] = (
                        self.inertia * particles_vel[idx] + cognitive_vel + social_vel
                    )
                    particles_vel[idx] = np.clip(
                        particles_vel[idx], -self.max_velocity, self.max_velocity
                    )
                    particles_pos[idx] += particles_vel[idx]
                    particles_pos[idx] = np.clip(
                        particles_pos[idx],
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
