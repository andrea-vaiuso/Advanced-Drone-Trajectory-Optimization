# Author: Andrea Vaiuso
# Version: 1.0
# Date: 06.08.2025
# Description: Class-based Grey Wolf Optimizer for PID gain tuning.
"""Grey Wolf Optimization for PID tuning packaged into a class."""

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


class GWOPIDOptimizer(Optimizer):
    """Optimize PID gains using the Grey Wolf Optimizer.

    Parameters
    ----------
    config_file : str, optional
        Path to the GWO configuration file.
    parameters_file : str, optional
        Path to the simulation parameters YAML file.
    verbose : bool, optional
        If ``True`` print step-by-step information.
    set_initial_obs : bool, optional
        Include current PID gains as the first wolf when ``True``.
    simulate_wind_flag : bool, optional
        Enable the Dryden wind model during simulations.
    waypoints : list, optional
        List of waypoints used for training. If ``None`` a default set is
        generated.
    """

    def __init__(
        self,
        config_file: str = "Settings/gwo_opt.yaml",
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
            "GWO",
            config_file,
            parameters_file,
            verbose=verbose,
            set_initial_obs=set_initial_obs,
            simulate_wind_flag=simulate_wind_flag,
            study_name=study_name,
            waypoints=waypoints,
            simulation_time=simulation_time,
        )

        gwo_cfg = self.cfg

        self.n_iter = int(gwo_cfg.get("n_iter", 100))
        self.pack_size = int(gwo_cfg.get("pack_size", 30))

        pbounds_cfg = gwo_cfg.get("pbounds", {})
        self.pbounds = {k: tuple(v) for k, v in pbounds_cfg.items()}
        self.lower_bounds = np.array([v[0] for v in self.pbounds.values()], dtype=float)
        self.upper_bounds = np.array([v[1] for v in self.pbounds.values()], dtype=float)
        self.dim = len(self.lower_bounds)

        self.costs: list[float] = []
        self.best_costs: list[float] = []

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def decode_wolf(self, vec: np.ndarray) -> Dict[str, tuple]:
        """Convert a wolf vector into a PID gain dictionary."""
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
        """Execute the Grey Wolf Optimization process."""
        rng = np.random.default_rng(42)
        pack = rng.uniform(
            self.lower_bounds, self.upper_bounds, size=(self.pack_size, self.dim)
        )

        if self.set_initial_obs:
            current_best = mainfunc.load_pid_gains(self.parameters)
            init_wolf = np.array(
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
            pack[0] = np.clip(init_wolf, self.lower_bounds, self.upper_bounds)

        alpha_pos = np.zeros(self.dim)
        beta_pos = np.zeros(self.dim)
        delta_pos = np.zeros(self.dim)
        alpha_cost = np.inf
        beta_cost = np.inf
        delta_cost = np.inf

        start_opt = time()
        print("Starting Grey Wolf Optimization...")
        try:
            for it in range(self.n_iter):
                for i in range(self.pack_size):
                    self.iteration = (i + 1) * (it + 1)
                    gains = self.decode_wolf(pack[i])
                    costs_sim = self.simulate_pid(gains)
                    total_cost = costs_sim["total_cost"]
                    self.costs.append(total_cost)
                    log_step(gains, total_cost, self.log_path, costs_sim)
                    if self.verbose:
                        print(
                            f"[ GWO ] Iteration {self.iteration}/{self.n_iter*self.pack_size} | "
                            f"cost={total_cost:.4f}, costs={costs_sim}"
                        )
                    if total_cost < alpha_cost:
                        delta_cost = beta_cost
                        delta_pos = beta_pos.copy()
                        beta_cost = alpha_cost
                        beta_pos = alpha_pos.copy()
                        alpha_cost = total_cost
                        alpha_pos = pack[i].copy()
                    elif total_cost < beta_cost:
                        delta_cost = beta_cost
                        delta_pos = beta_pos.copy()
                        beta_cost = total_cost
                        beta_pos = pack[i].copy()
                    elif total_cost < delta_cost:
                        delta_cost = total_cost
                        delta_pos = pack[i].copy()
                    self.best_costs.append(alpha_cost)

                a = 2 - it * (2 / self.n_iter)
                for i in range(self.pack_size):
                    for d in range(self.dim):
                        r1, r2 = rng.random(2)
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[d] - pack[i, d])
                        X1 = alpha_pos[d] - A1 * D_alpha

                        r1, r2 = rng.random(2)
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[d] - pack[i, d])
                        X2 = beta_pos[d] - A2 * D_beta

                        r1, r2 = rng.random(2)
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[d] - pack[i, d])
                        X3 = delta_pos[d] - A3 * D_delta

                        pack[i, d] = np.clip(
                            (X1 + X2 + X3) / 3,
                            self.lower_bounds[d],
                            self.upper_bounds[d],
                        )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
        finally:
            tot_time = time() - start_opt
            if np.isinf(alpha_cost):
                print("No evaluations were performed.")
                return
            best_params = self.decode_wolf(alpha_pos)
            show_best_params(
                "Grey Wolf Optimization",
                self.parameters,
                best_params,
                self.opt_output_path,
                alpha_cost,
                self.iteration,
                self.simulation_time,
                tot_time,
            )
            plot_costs_trend(
                self.costs,
                save_path=self.opt_output_path.replace(".txt", "_costs.png"),
                alg_name="Grey Wolf Optimization",
            )
            plot_costs_trend(
                self.best_costs,
                save_path=self.opt_output_path.replace(".txt", "_best_costs.png"),
                alg_name="Grey Wolf Optimization",
            )


def main() -> None:
    """Run PID optimization using the Grey Wolf Optimizer."""
    optimizer = GWOPIDOptimizer(
        config_file="Settings/gwo_opt.yaml",
        parameters_file="Settings/simulation_parameters.yaml",
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
        waypoints=mainfunc.create_training_waypoints(),
    )
    optimizer.optimize()


if __name__ == "__main__":
    main()

