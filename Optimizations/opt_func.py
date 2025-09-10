"""Shared utility functions for optimization scripts."""

import json
from datetime import datetime
from time import time
from typing import Dict, Any
import matplotlib.pyplot as plt

import numpy as np
from Drone.Simulation import Simulation

# timers used for logging
start_time = time()
last_time = start_time


def plot_costs_trend(costs: list, save_path: str = None, alg_name: str = "") -> None:
    """
    Plot the trend of costs over iterations. If `save_path` is provided, save the plot to that path.
    A red marker indicates the best cost found, while a blue line shows the trend.
    """
    # Make all costs positive for better visualization
    costs = [abs(cost) for cost in costs]

    plt.figure(figsize=(10, 5))
    plt.plot(costs, color='blue', label='Cost Trend', marker='o', markersize=3, zorder=1)
    best_cost = min(costs)
    best_idx = costs.index(best_cost)
    plt.scatter(best_idx, best_cost, color='red', label='Best Cost', zorder=5, marker='x', s=100)
    plt.title(f'Cost Trend Over Iterations - ({alg_name})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def log_step(params: Dict[str, Any], cost: float, log_path: str, costs: dict = None) -> None:
    """Append a single optimization step to a JSON log file."""
    global last_time
    current = time()
    entry = {
        'target': -cost,
        'params': {
            k: ([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v))
            for k, v in params.items()
        },
        'datetime': {
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed': current - start_time,
            'delta': current - last_time,
        },
        'costs': costs if costs is not None else {},
    }
    with open(log_path, 'a') as f:
        json.dump(entry, f)
        f.write('\n')
    last_time = current



def calculate_costs(sim: Simulation, simulation_time: float,
                    pitch_roll_oscillation_weight: float = 1.0,
                    thrust_oscillation_weight: float = 1e-5,
                    altitude_weight: float = 1.0,
                    power_weight: float = 1e-4,
                    noise_weight: float = 2e-25,
                    print_costs: bool = False) -> tuple:

    """Compute the cost metrics used for PID gain optimization."""
    angles = np.array(sim.angles_history)
    final_time = sim.navigation_time if sim.navigation_time is not None else simulation_time

    final_target = {
        'x': sim.waypoints[-1]['x'],
        'y': sim.waypoints[-1]['y'],
        'z': sim.waypoints[-1]['z'],
    }
    final_distance = np.linalg.norm(
        sim.drone.state['pos'] - np.array([final_target['x'], final_target['y'], final_target['z']])
    )

    power_cost = np.sum(np.array(sim.power_history)) * power_weight

    altitude_rule_cost = np.sum(np.array(sim.altitude_penalty_history)) * altitude_weight

    time_cost = final_time

    p = 12  # norm order for noise cost
    if len(sim.swl_history) > 0:
        swl = np.array(sim.swl_history, dtype=float)
        swl_weighted = swl * np.array(sim.noise_penalty_history)
        noise_cost = noise_weight * (np.linalg.norm(swl_weighted, ord=p)**p + np.max(swl))
    else:
        noise_cost = 0.0

    total_cost = time_cost + \
        power_cost + \
        altitude_rule_cost + \
        noise_cost

    if not sim.has_moved:
        total_cost += 1000

    result = {
        'total_cost': total_cost,
        'time_cost': time_cost,
        'power_cost': power_cost,
        'noise_cost': noise_cost,
        'n_waypoints_completed': sim.current_seg_idx,
        'tot_waypoints': len(sim.waypoints),
    }

    if print_costs: print(result)
    return result

def run_simulation(pid_gains: Dict[str, tuple],
                   parameters: dict,
                   waypoints: list,
                   world,
                   thrust_max: float,
                   simulation_time: float,
                   noise_model=None,
                   simulate_wind: bool = False) -> Dict[str, float]:
    """Execute a simulation

    Parameters
    ----------
    pid_gains : Dict[str, tuple]
        PID gains for the controller.
    parameters : dict
        Simulation parameters loaded from a YAML file.
    waypoints : list
        Waypoints followed by the drone during the simulation.
    world : World
        Environment in which the simulation takes place.
    thrust_max : float
        Maximum thrust obtainable from the rotor model.
    simulation_time : float
        Maximum duration of the simulation in seconds.
    noise_model : optional
        Acoustic noise model used in the simulation, ``None`` disables noise
        calculations.
    simulate_wind : bool, optional
        If ``True``, enable the Dryden wind model during the simulation.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the cost metrics returned by :func:`calculate_costs`.
    """

    # Import here to avoid circular imports when ``main`` imports this module
    import main as mainfunc  # noqa: WPS433 (acceptable in this context)

    init_state = mainfunc.create_initial_state()

    quad_controller = mainfunc.create_quadcopter_controller(
        init_state=init_state,
        pid_gains=pid_gains,
        t_max=thrust_max,
        parameters=parameters,
    )
    drone = mainfunc.create_quadcopter_model(
        init_state=init_state, quad_controller=quad_controller, parameters=parameters
    )
    # sim = mainfunc.create_simulation(
    #     drone=drone,
    #     world=world,
    #     waypoints=waypoints,
    #     parameters=parameters,
    #     noise_model=noise_model,
    #     generate_sound_map=False,
    # )
    sim = Simulation(
        drone,
        world,
        waypoints, 
        dt=float(parameters['dt']),
        max_simulation_time=float(parameters['simulation_time']),
        frame_skip=int(parameters['frame_skip']),
        target_reached_threshold=float(parameters['threshold']),
        target_shift_threshold_distance=float(parameters['target_shift_threshold_distance']),
        noise_annoyance_radius=int(parameters['noise_annoyance_radius']),
        noise_model=noise_model,
        generate_sound_emission_map=False,
        compute_psychoacoustics=False,
    )
    if simulate_wind:
        sim.setWind(
            max_simulation_time=simulation_time,
            dt=float(parameters["dt"]),
            height=100,
            airspeed=10,
            turbulence_level=10,
            plot_wind_signal=False,
            seed=None,
        )
    sim.startSimulation(
        stop_at_target=True, verbose=False, stop_sim_if_not_moving=True, use_static_target=True
    )
    return calculate_costs(sim, simulation_time)


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to a ``HH:MM:SS`` formatted string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def show_best_params(
    alg_name,
    sim_parameters,
    best_params,
    opt_output_path,
    global_best_cost,
    n_iter,
    simulation_time,
    tot_time,
):
    print("Best trajectory found:")
    for i, wp in enumerate(best_params):
        print("WP{}: [{:.5g}, {:.5g}, {:.5g}]".format(i, wp["x"], wp["y"], wp["z"]))
    print("Best cost value: {:.4f}".format(global_best_cost))

    with open(opt_output_path, "w") as f:
        f.write(f"Optimization Algorithm: {alg_name}\n")
        f.write("Best trajectory found:\n")
        for i, wp in enumerate(best_params):
            f.write(f"WP{i}: ({wp['x']}, {wp['y']}, {wp['z']})\n")
        f.write(f"Best cost value: {global_best_cost}\n")
        f.write(f"Total optimization time: {seconds_to_hhmmss(tot_time)}\n")
        f.write(f"N iterations: {n_iter}\n")
        f.write(f"Max sim time: {simulation_time:.2f} seconds\n")
        f.write(f"Avg step time: {tot_time / n_iter:.2f} seconds\n")
        f.write("\n----------------\nSimulation parameters:\n")
        for k, v in sim_parameters.items():
            f.write(f"{k}: {v}\n")

