"""Shared utilities for the metaheuristic waypoint optimizers."""

import json
import os
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

from Drone.Simulation import Simulation


class MetaHeuristicOptimizer:
    """Base class offering logging, configuration and evaluation helpers.

    Subclasses only need to implement :meth:`optimize` while relying on this
    base for configuration loading, cost computation, JSON logging and plotting
    convenience.  The class keeps track of the simulation object so optimizers
    can trigger rollouts without reimplementing boilerplate.
    """

    def __init__(
        self,
        simulation_object: Simulation,
        opt_method_name: str,
        config_file=None,
        verbose: bool = True,
        set_initial_obs: bool = True,
        study_name: str = "",
    ) -> None:
        """Initialise the optimizer façade.

        Args:
            simulation_object: The simulator instance shared by all optimizers.
            opt_method_name: Human readable algorithm identifier.
            config_file: Path to the YAML configuration file or ``None`` to use
                the defaults coded in subclasses.
            verbose: When ``True`` prints progress messages.
            set_initial_obs: Whether optimizers may inject a deterministic
                starting particle as the first individual of the population.
            study_name: Optional suffix appended to the output directory name.
        """

        self.opt_method_name = opt_method_name
        if config_file is not None:
            print(f"{self.get_alg_prefix()} Loading configuration from {config_file}")
            with open(config_file, "r", encoding="utf-8") as stream:
                self.cfg = yaml.safe_load(stream) or {}
        else:
            print(
                f"{self.get_alg_prefix()} No configuration file provided, using default settings."
            )
            self.cfg = {}

        self.simulation_object = simulation_object
        self.verbose = verbose
        self.set_initial_obs = set_initial_obs
        self.study_name = study_name

        self.base_dir = os.path.join("Optimizations", opt_method_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.study_dir = None

        self.costs_history = []
        self.costs_dict_history = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.last_time = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_alg_prefix(self) -> str:
        """Return a prefix used in log messages for readability."""

        return f"[ {self.opt_method_name} ]>"

    def start_optimization(self) -> None:
        """Execute :meth:`optimize` and persist the resulting study artefacts."""

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = datetime_str if not self.study_name else f"{datetime_str}_{self.study_name}"
        self.study_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(self.study_dir, exist_ok=True)

        self.start_time = time()
        best_params = self.optimize()
        self.end_time = time() - self.start_time

        if self.verbose:
            print(
                f"{self.get_alg_prefix()} Optimization completed in {self.end_time:.2f} seconds."
            )

        self.save_results_in_file(best_params)
        self.plot_costs_trend(save_fig=True)

    def optimize(self):
        """Run the optimisation routine and return the best trajectory found."""

        raise NotImplementedError("Subclasses must implement `optimize`.")

    def plot_costs_trend(self, save_fig: bool = True) -> None:
        """Plot and optionally save the running best cost across iterations."""

        if not self.costs_history:
            return

        costs = [abs(cost) for cost in self.costs_history]
        best_costs = [min(costs[: idx + 1]) for idx in range(len(costs))]

        plt.figure(figsize=(10, 5))
        plt.plot(best_costs, color="blue", label="Cost Trend", marker="o", markersize=3, zorder=1)
        best_cost = min(costs)
        best_idx = costs.index(best_cost)
        plt.scatter(best_idx, best_cost, color="red", label="Best Cost", zorder=5, marker="x", s=100)
        plt.title(f"Cost Trend Over Iterations - ({self.opt_method_name})")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend()
        if save_fig and self.study_dir:
            path = os.path.join(self.study_dir, "costs_trend.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"{self.get_alg_prefix()} Cost trend plot saved to {path}")
        plt.show()

    def log_step(self, opt_params) -> None:
        """Append a single optimisation step to the JSON log.

        Args:
            opt_params: Absolute waypoint dictionaries evaluated in the step.
        """

        if not self.study_dir:
            return

        current_time = time()
        current_cost = self.costs_history[-1] if self.costs_history else None
        entry = {
            "target": current_cost,
            "best_target": min(self.costs_history) if self.costs_history else current_cost,
            "params": list(opt_params),
            "datetime": {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed": current_time - self.start_time,
                "delta": current_time - self.last_time,
            },
            "costs": self.costs_dict_history[-1] if self.costs_dict_history else {},
        }
        with open(os.path.join(self.study_dir, "optimization_history_log.json"), "a", encoding="utf-8") as file:
            json.dump(entry, file)
            file.write("\n")
        self.last_time = current_time

    # ------------------------------------------------------------------
    # Cost utilities
    # ------------------------------------------------------------------
    def calculate_costs(
        self,
        time_weight: float = 1.0,
        altitude_weight: float = 1e-2,
        power_weight: float = 1e-4,
        noise_weight: float = 1.5e1,
        completion_weight: float = 1000.0,
        print_costs: bool = False,
        weight_penalties: bool = True,
        save_costs_in_history: bool = True,
    ):
        """Aggregate the simulation penalties into a scalar cost.

        Args:
            altitude_weight: Multiplier applied to altitude rule violations.
            power_weight: Multiplier applied to the total energy consumption.
            noise_weight: Weight applied to psychoacoustic annoyance metrics.
            completion_weight: Penalty encouraging full route completion.
            print_costs: When ``True`` prints the resulting dictionary.
            weight_penalties: Toggle for enforcing the additional penalties.
            save_costs_in_history: When ``True`` append the result to
                :attr:`costs_history` for later analysis.

        Returns:
            Dictionary with each cost component and the aggregated ``total_cost``.
        """

        sim = self.simulation_object
        noise_penalty_history = np.array(sim.noise_penalty_history)
        final_time = sim.navigation_time if sim.navigation_time is not None else sim.max_simulation_time

        power_cost = np.sum(np.array(sim.power_history)) * power_weight
        altitude_rule_cost = (
            np.sum(np.array(sim.altitude_penalty_history)) * altitude_weight if weight_penalties else 0.0
        )

        time_cost = final_time * time_weight
        if sim.spl_history:
            spl = np.array(sim.spl_history, dtype=float)
            spl_weighted = spl * noise_penalty_history if weight_penalties else spl
            noise_cost = np.mean(spl_weighted) * noise_weight
        else:
            noise_cost = 0.0

        perc_completed = sim.current_seg_idx / len(sim.waypoints) if sim.waypoints else 0.0
        complet_cost = completion_weight * (1 - perc_completed)

        total_cost = time_cost + power_cost + noise_cost + altitude_rule_cost + complet_cost
        result = {
            "total_cost": float(total_cost),
            "time_cost": float(time_cost),
            "power_cost": float(power_cost),
            "noise_cost": float(noise_cost),
            "completion_cost": float(complet_cost),
            "n_waypoints_completed": int(sim.current_seg_idx),
            "tot_waypoints": len(sim.waypoints),
        }

        if save_costs_in_history:
            self.costs_history.append(float(total_cost))
            self.costs_dict_history.append(result)

        if print_costs:
            print(result)
        return result

    def run_simulation(self, waypoints_set):
        """Execute the simulator on the provided waypoint list."""

        self.simulation_object.waypoints = list(waypoints_set)
        self.simulation_object.startSimulation(
            stop_at_target=True,
            verbose=False,
            stop_sim_if_not_moving=True,
            use_static_target=False,
            reset_drone_state=True,
        )
        return self.calculate_costs()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_results_in_file(self, best_params) -> None:
        """Persist the summary of an optimisation run to disk."""

        if not self.study_dir:
            return

        results = {
            "Optimization Algorithm": self.opt_method_name,
            "Best Parameters Found": list(best_params),
            "Best Cost": min(self.costs_history) if self.costs_history else None,
            "Total time (s)": self.end_time,
            "Number of iterations": len(self.costs_history),
            "Avg step time (s)": self.end_time / len(self.costs_history) if self.costs_history else None,
            "Optimization Parameters": self.cfg,
        }
        path = os.path.join(self.study_dir, "optimization_results.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"{self.get_alg_prefix()} Results saved to {path}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def decode_particle(particle: np.ndarray, n_points: int, A, B):
        """Convert a particle vector into absolute waypoint dictionaries."""

        pts = n_points + 2
        default_wp = []
        for i in range(1, pts - 1):
            alpha = i / (pts - 1)
            x = A[0] + alpha * (B[0] - A[0])
            y = A[1] + alpha * (B[1] - A[1])
            z = A[2] + alpha * (B[2] - A[2])
            default_wp.append((x, y, z))

        waypoints = []
        for i in range(n_points):
            idx = 4 * i
            dx, dy, dz, v = particle[idx : idx + 4]
            x0, y0, z0 = default_wp[i]
            waypoints.append({"x": x0 + dx, "y": y0 + dy, "z": z0 + dz, "v": float(v)})

        waypoints.append({"x": float(B[0]), "y": float(B[1]), "z": float(B[2]), "v": 5.0})
        return waypoints

    @staticmethod
    def linspace_internal_points(A, B, n_points: int) -> np.ndarray:
        """Return the neutral reference points along the A→B line segment."""

        pts = n_points + 2
        default_wp = []
        for i in range(1, pts - 1):
            alpha = i / (pts - 1)
            x = A[0] + alpha * (B[0] - A[0])
            y = A[1] + alpha * (B[1] - A[1])
            z = A[2] + alpha * (B[2] - A[2])
            default_wp.append((x, y, z))
        return np.array(default_wp, dtype=float)

    @staticmethod
    def build_particle_bounds(
        A,
        B,
        n_points: int,
        max_perturbation_offset,
        vmax: float,
        world_min: float = 0.0,
        world_max: float = 1000.0,
        v_min: float = 5.0,
    ):
        """Compute the feasible perturbation bounds for each waypoint."""

        base = MetaHeuristicOptimizer.linspace_internal_points(A, B, n_points)
        dmin = np.array([world_min, world_min, world_min]) - base
        dmax = np.array([world_max, world_max, world_max]) - base

        if max_perturbation_offset is not None:
            cap = np.full(3, max_perturbation_offset)
            dmin = np.maximum(dmin, -cap)
            dmax = np.minimum(dmax, cap)

        lows = []
        highs = []
        for i in range(n_points):
            lows.extend([dmin[i, 0], dmin[i, 1], dmin[i, 2], v_min])
            highs.extend([dmax[i, 0], dmax[i, 1], dmax[i, 2], float(vmax)])
        return np.array(lows, dtype=float), np.array(highs, dtype=float)
