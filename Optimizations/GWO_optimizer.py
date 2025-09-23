from Optimizations.optimizer import MetaHeuristicOptimizer
from Drone.Simulation import Simulation
import numpy as np

class GWOOptimizer(MetaHeuristicOptimizer):
    """Grey Wolf Optimizer (GWO) implementation."""

    def __init__(self, simulation_object: Simulation, 
                config_file: str,
                A: tuple,
                B: tuple,
                max_perturbation_offset,
                n_points: int = 5,
                n_generations: int = 100,
                pack_size: int = 30,
                max_velocity: float = 20,
                verbose: bool = True):
        super().__init__(simulation_object=simulation_object, 
                         opt_method_name="GWO", 
                         config_file=config_file, 
                         verbose=verbose)
        self.n_generations = n_generations
        self.pack_size = pack_size
        self.n_points = n_points
        self.max_perturbation_offset = max_perturbation_offset
        self.max_velocity = max_velocity
        self.A = A
        self.B = B
        self.global_best_cost = np.inf
        self.global_best_pos = None

    def optimize(self, seed=42):

        low_bounds, high_bounds = self.build_particle_bounds(
            self.A, self.B, self.n_points, 
            self.max_perturbation_offset, 
            self.max_velocity, 
            world_min=0.0, world_max=1000.0
        )

        rng = np.random.default_rng(seed)
        wolves_pos = rng.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(self.pack_size, self.n_points * 4)
        )
        if self.set_initial_obs:
            init_wolf = np.array([
                    0.0, 0.0, 0.0, 5.0
                ] * self.n_points
            )
            wolves_pos[0] = init_wolf
        else:
            # Random initialization within bounds and max velocity
            init_wolf = np.random.uniform(low=self.A, high=self.B, size=(self.n_points, 4))
            init_wolf[:, 3] = np.random.uniform(0, self.max_velocity, size=self.n_points)

        personal_best_pos = wolves_pos.copy()
        personal_best_cost = np.full(self.pack_size, np.inf)

        try:
            for gen in range(self.n_generations):
                for i in range(self.pack_size):
                    waypoints = self.decode_particle(wolves_pos[i], self.n_points, self.A, self.B)
                    costs_sim = self.run_simulation(waypoints)
                    total_cost = costs_sim["total_cost"]
                    self.log_step(waypoints)
                    if total_cost < personal_best_cost[i]:
                        personal_best_cost[i] = total_cost
                        personal_best_pos[i] = wolves_pos[i].copy()
                    if total_cost < self.global_best_cost:
                        self.global_best_cost = total_cost
                        self.global_best_pos = wolves_pos[i].copy()
                a = 2 - gen * (2 / self.n_generations)  # a decreases linearly from 2 to 0
                sorted_indices = np.argsort(personal_best_cost)
                alpha_pos = personal_best_pos[sorted_indices[0]]
                beta_pos = personal_best_pos[sorted_indices[1]]
                delta_pos = personal_best_pos[sorted_indices[2]]
                for i in range(self.pack_size):
                    for j in range(self.n_points * 4):
                        r1, r2 = rng.random(), rng.random()
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[j] - wolves_pos[i][j])
                        X1 = alpha_pos[j] - A1 * D_alpha
                        r1, r2 = rng.random(), rng.random()
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[j] - wolves_pos[i][j])
                        X2 = beta_pos[j] - A2 * D_beta
                        r1, r2 = rng.random(), rng.random()
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[j] - wolves_pos[i][j])
                        X3 = delta_pos[j] - A3 * D_delta
                        wolves_pos[i][j] = (X1 + X2 + X3) / 3
                    wolves_pos[i] = np.clip(wolves_pos[i], 
                                           low_bounds, 
                                           high_bounds)
                if self.verbose:
                    print(
                        f"{self.get_alg_prefix()} Gen {gen+1}/{self.n_generations}, Best Cost: {self.global_best_cost:.2f}, best: {min(self.costs_history):.2f}"
                    )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
        finally:
            if self.global_best_pos is None:
                print(f"{self.get_alg_prefix()} No valid solution found.")
                return
            
            best_params = self.decode_particle(self.global_best_pos, self.n_points, self.A, self.B)
            return best_params