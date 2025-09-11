from Drone.Simulation import Simulation
from datetime import datetime
from time import time
import yaml
import os
import matplotlib.pyplot as plt
import json
import numpy as np

class MetaHeuristicOptimizer():
    """A meta-optimizer class to manage different optimization algorithms."""

    def __init__(self,
                simulation_object: Simulation,
                opt_method_name: str,
                config_file: str,
                verbose: bool = True,
                set_initial_obs: bool = True,
                study_name: str = "") -> None:
        self.opt_method_name = opt_method_name
        if config_file is not None:
            print(f"{self.get_alg_prefix()} Loading configuration from {config_file}")
            with open(config_file, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            print(f"{self.get_alg_prefix()} No configuration file provided, using default settings.")
            self.cfg = {}

        self.last_time = 0

        self.simulation_object = simulation_object
        
        self.verbose = verbose
        self.set_initial_obs = set_initial_obs
        self.study_name = study_name
        self.base_dir = os.path.join("Optimizations", opt_method_name)
        self.study_dir = None
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.costs_history = []
        self.costs_dict_history = []

    def get_alg_prefix(self) -> str:
        return f"[ {self.opt_method_name} ]>"

    def start_optimization(self):
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = datetime_str if not self.study_name else f"{datetime_str}_{self.study_name}"
        self.study_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(self.study_dir, exist_ok=True)
        self.start_time = time()
        self.optimize()
        self.end_time = time() - self.start_time
        if self.verbose:
            print(f"{self.get_alg_prefix()} Optimization completed in {self.end_time:.2f} seconds.")
    
    def optimize(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def plot_costs_trend(self, save_fig: bool = True) -> None:
        """
        Plot the trend of costs over iterations. If `save_path` is provided, save the plot to that path.
        A red marker indicates the best cost found, while a blue line shows the trend.
        """
        # Make all costs positive for better visualization
        costs = [abs(cost) for cost in self.costs_history]
        best_costs = []
        for i in range(len(costs)):
            best_costs.append(min(costs[:i+1]))
        
        plt.figure(figsize=(10, 5))
        plt.plot(best_costs, color='blue', label='Cost Trend', marker='o', markersize=3, zorder=1)
        best_cost = min(costs)
        best_idx = costs.index(best_cost)
        plt.scatter(best_idx, best_cost, color='red', label='Best Cost', zorder=5, marker='x', s=100)
        plt.title(f'Cost Trend Over Iterations - ({self.opt_method_name})')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        if save_fig:
            plt.savefig(os.path.join(self.study_dir, "costs_trend.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def log_step(self, opt_params: list) -> None:
        """Append a single optimization step to a JSON log file.

        Parameters:
        -----------
        opt_params : list
            The parameters used in the current optimization step.
        """
        current_time = time()
        current_cost = self.costs_history[-1] if self.costs_history else None
        entry = {
            'target': current_cost,
            'best_target': min(self.costs_history) if self.costs_history else current_cost,
            'params': opt_params,
            'datetime': {
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed': current_time - self.start_time,
                'delta': current_time - self.last_time,
            },
            'costs': self.costs_dict_history[-1] if self.costs_dict_history else {},
        }
        with open(os.path.join(self.study_dir, "optimization_history_log.json"), 'a') as f:
            json.dump(entry, f)
            f.write('\n')
        self.last_time = current_time

    def calculate_costs(self,
                        altitude_weight: float = 1.0,
                        power_weight: float = 1e-4,
                        noise_weight: float = 2e-25,
                        completion_weight: float = 1000.0,
                        print_costs: bool = False,
                        weight_penalties: bool = True,
                        save_costs_in_history: bool = True
                        ) -> tuple:
        noise_penalty_history = np.array(self.simulation_object.noise_penalty_history)
        final_time = self.simulation_object.navigation_time if self.simulation_object.navigation_time is not None else self.simulation_object.max_simulation_time
        power_cost = np.sum(np.array(self.simulation_object.power_history)) * power_weight
        if weight_penalties:
            altitude_rule_cost = np.sum(np.array(self.simulation_object.altitude_penalty_history)) * altitude_weight
        else:
            altitude_rule_cost = 0
        time_cost = final_time
        p = 12  # norm order for noise cost
        if self.simulation_object.swl_history:
            swl = np.array(self.simulation_object.swl_history, dtype=float)
            if weight_penalties:
                # Elementwise multiply the sound power levels by the noise penalties
                swl_weighted = swl * noise_penalty_history
            else:
                swl_weighted = swl
            noise_cost = noise_weight * (np.linalg.norm(swl_weighted, ord=p)**p + np.max(swl))
        else:
            noise_cost = 0.0

        perc_completed = self.simulation_object.current_seg_idx / len(self.simulation_object.waypoints)
        complet_cost = completion_weight * (1 - perc_completed)

        total_cost = time_cost + power_cost + noise_cost + altitude_rule_cost + complet_cost

        result = {
            'total_cost': total_cost,
            'time_cost': time_cost,
            'power_cost': power_cost,
            'noise_cost': noise_cost,
            'completion_cost': complet_cost,
            'n_waypoints_completed': self.simulation_object.current_seg_idx,
            'tot_waypoints': len(self.simulation_object.waypoints),
        }
        if save_costs_in_history:
            self.costs_history.append(total_cost)
            self.costs_dict_history.append(result)

        if print_costs: print(result)
        return result

    def run_simulation(self, waypoints_set) -> dict:
        """
        Run the simulation and return the cost metrics.

        Returns:
        --------
        dict
            A dictionary containing the computed cost metrics.
        """
        self.simulation_object.waypoints = waypoints_set

        # if self.simulation_object.simulate_wind:
        #     self.simulation_object.setWind(
        #         max_simulation_time=self.simulation_object.simulation_time,
        #         dt=float(self.simulation_object.dt),
        #         height=100,
        #         airspeed=10,
        #         turbulence_level=10,
        #         plot_wind_signal=False,
        #         seed=None,
        #     )
        self.simulation_object.startSimulation(
            stop_at_target=True, verbose=False, 
            stop_sim_if_not_moving=True, use_static_target=False,
            reset_drone_state=True
        )
        return self.calculate_costs()
    
    def save_results_in_file(self, best_params) -> None:
        """Save the optimization results to a JSON file."""
        results = {
            'Optimization Algorithm': self.opt_method_name,
            'Best Parameters Found': best_params,
            'Best Cost': min(self.costs_dict_history),
            'Total time (s)': self.end_time,
            'Number of iterations': len(self.costs_history),
            'Avg step time (s)': self.end_time / len(self.costs_history) if self.costs_history else None,
            'Optimization Parameters': self.cfg
        }
        with open(os.path.join(self.study_dir, "optimization_results.json"), 'w') as f:
            json.dump(results, f, indent=4)