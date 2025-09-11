from Optimizations.optimizer import MetaHeuristicOptimizer
from Drone.Simulation import Simulation
import numpy as np

class PSOOptimizer(MetaHeuristicOptimizer):
    """Particle Swarm Optimization (PSO) implementation."""

    def __init__(self, simulation_object: Simulation, 
                config_file: str,
                n_generations: int = 100,
                swarm_size: int = 30,
                n_points: int = 5,
                max_perturbation_offset: float = 500,
                max_velocity: float = 20,
                A: tuple = (0, 0, 0),
                B: tuple = (1000, 1000, 1000),
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

    def decode_particle(self, particle: np.ndarray) -> dict:
        """Decode a particle array (list of perturbation respect to default point positions) into a list of waypoints"""
        perturbations = []
        for i in range(self.n_points):
            idx = i * 4
            # waypoint: x, y, z, vel
            p = (particle[idx], particle[idx + 1], particle[idx + 2], particle[idx + 3])
            perturbations.append(p)
        # Build the default set of waypoints: equally distributed between A and B
        default_waypoints = []
        pts = self.n_points+2
        for i in range(pts):
            alpha = i / (pts - 1) if pts > 1 else 0
            x = self.A[0] + alpha * (self.B[0] - self.A[0])
            y = self.A[1] + alpha * (self.B[1] - self.A[1])
            z = self.A[2] + alpha * (self.B[2] - self.A[2])
            vel = 5.0
            default_waypoints.append((x, y, z, vel))
        # Remove first and last (A and B)
        default_waypoints = default_waypoints[1:-1]
        # Apply perturbations to default waypoints
        waypoints = []
        for i in range(self.n_points):
            wp = default_waypoints[i]
            perturbation = perturbations[i]
            # Apply perturbation: (x, y, z, vel) + (dx, dy, dz, dvel)
            new_wp = {
                'x': wp[0] + perturbation[0],
                'y': wp[1] + perturbation[1],
                'z': wp[2] + perturbation[2],
                'v': perturbation[3],
            }
            waypoints.append(new_wp)
        waypoints.append({'x': self.B[0], 'y': self.B[1], 'z': self.B[2], 'v': 5.0})  # Ensure the last waypoint is B
        return waypoints

    def optimize(self):
        rng = np.random.default_rng(42)
        particles_pos = rng.uniform(
            low=[0, 0, 0, 0.0] * self.n_points,
            high=[self.max_perturbation_offset, self.max_perturbation_offset, self.max_perturbation_offset, self.max_velocity] * self.n_points,
            size=(self.swarm_size, self.n_points * 4)
        )
        particles_vel = np.zeros_like(particles_pos)
        if self.set_initial_obs:
            init_particle = np.array([
                    0.0, 0.0, 0.0, 5.0
                ] * self.n_points
            )
            particles_pos[0] = np.clip(init_particle, 
                                       [0, 0, 0, 0.0] * self.n_points, 
                                       [self.max_perturbation_offset, 
                                        self.max_perturbation_offset, 
                                        self.max_perturbation_offset, 
                                        self.max_velocity] * self.n_points)
        else:
            # Random initialization within bounds and max velocity
            init_particle = np.random.uniform(low=self.A, high=self.B, size=(self.n_points, 4))
            init_particle[:, 3] = np.random.uniform(0, self.max_velocity, size=self.n_points)

        personal_best_pos = particles_pos.copy()
        personal_best_cost = np.full(self.swarm_size, np.inf)

        try:
            for gen in range(self.n_generations):
                for i in range(self.swarm_size):
                    waypoints = self.decode_particle(particles_pos[i])
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
                        [0, 0, 0, 0.0] * self.n_points,
                        [self.max_perturbation_offset, self.max_perturbation_offset, self.max_perturbation_offset, self.max_velocity] * self.n_points
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

            best_params = self.decode_particle(self.global_best_pos)
            self.save_results_in_file(best_params)
            self.plot_costs_trend(save_fig=True)
            return best_params

