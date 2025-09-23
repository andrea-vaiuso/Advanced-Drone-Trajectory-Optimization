from Optimizations.optimizer import MetaHeuristicOptimizer
from Drone.Simulation import Simulation
import numpy as np

class PSOOptimizer(MetaHeuristicOptimizer):
    """Particle Swarm Optimization (PSO) implementation."""

    def __init__(self, simulation_object: Simulation, 
                config_file: str,
                A: tuple,
                B: tuple,
                max_perturbation_offset,
                n_points,
                n_generations: int = 100,
                swarm_size: int = 30,
                max_velocity: float = 20,
                verbose: bool = True):
        super().__init__(simulation_object=simulation_object, 
                         opt_method_name="PSO", 
                         config_file=config_file, 
                         verbose=verbose)
        self.n_generations = n_generations
        self.swarm_size = swarm_size
        self.n_points = n_points
        self.max_perturbation_offset = max_perturbation_offset
        self.max_velocity = max_velocity
        self.A = A
        self.B = B
        self.inertia = self.cfg.get("inertia", 0.5)
        self.cognitive_coeff = self.cfg.get("cognitive_coeff", 1.5)
        self.social_coeff = self.cfg.get("social_coeff", 1.5)
        self.global_best_cost = np.inf
        self.global_best_pos = None

    def optimize(self, seed=42):

        low_bounds, high_bounds = self.build_particle_bounds(
            self.A, self.B, self.n_points, 
            self.max_perturbation_offset, 
            self.max_velocity, 
            world_min=0, world_max=self.simulation_object.world.max_world_size,
        )

        rng = np.random.default_rng(seed)
        particles_pos = rng.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(self.swarm_size, self.n_points * 4)
        )
        particles_vel = np.zeros_like(particles_pos)
        if self.set_initial_obs:
            init_particle = np.array([
                    0.0, 0.0, 0.0, 5.0
                ] * self.n_points
            )
            particles_pos[0] = np.clip(init_particle, 
                                       low_bounds, 
                                       high_bounds)
        else:
            # Random initialization within bounds and max velocity
            init_particle = np.random.uniform(low=self.A, high=self.B, size=(self.n_points, 4))
            init_particle[:, 3] = np.random.uniform(0, self.max_velocity, size=self.n_points)

        personal_best_pos = particles_pos.copy()
        personal_best_cost = np.full(self.swarm_size, np.inf)

        try:
            for gen in range(self.n_generations):
                for i in range(self.swarm_size):
                    waypoints = self.decode_particle(particles_pos[i], self.n_points, self.A, self.B)
                    costs_sim = self.run_simulation(waypoints)
                    total_cost = costs_sim["total_cost"]
                    self.log_step(waypoints)
                    if self.verbose:
                        print(
                            f"{self.get_alg_prefix()} Gen {gen+1}/{self.n_generations}, Particle {i+1}/{self.swarm_size}, Cost: {costs_sim['total_cost']:.2f}, best: {min(self.costs_history):.2f}"
                        )
                    if total_cost < personal_best_cost[i]:
                        personal_best_cost[i] = total_cost
                        personal_best_pos[i] = particles_pos[i].copy()
                    if total_cost < self.global_best_cost:
                        self.global_best_cost = total_cost
                        self.global_best_pos = particles_pos[i].copy()
                for i in range(self.swarm_size):
                    r1, r2 = rng.random(2)
                    cognitive_vel = self.cognitive_coeff * r1 * (personal_best_pos[i] - particles_pos[i])
                    social_vel = self.social_coeff * r2 * (self.global_best_pos - particles_pos[i])
                    particles_vel[i] = (self.inertia * particles_vel[i] + cognitive_vel + social_vel)
                    particles_vel[i] = np.clip(particles_vel[i], -self.max_velocity, self.max_velocity)
                    particles_pos[i] += particles_vel[i]
                    particles_pos[i] = np.clip(
                        particles_pos[i],
                        low_bounds,
                        high_bounds
                    )
        except KeyboardInterrupt:
            print(f"{self.get_alg_prefix()} Optimization interrupted by user.")
        except Exception as e:
            print(f"{self.get_alg_prefix()} An error occurred during optimization: {e.__traceback__}")
            raise e
        finally:
            if self.global_best_cost is None or self.global_best_pos is None:
                print(f"{self.get_alg_prefix()} No valid solution found.")
                return

            best_params = self.decode_particle(self.global_best_pos, self.n_points, self.A, self.B)
            return best_params

