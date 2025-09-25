"""Gymnasium environment for drone trajectory optimization with SAC."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from Drone.Simulation import Simulation
from Optimizations.optimizer import MetaHeuristicOptimizer


class CostEvaluator(MetaHeuristicOptimizer):
    """Lightweight wrapper to reuse the metaheuristic cost function in RL."""

    def __init__(self, simulation_object: Simulation, name: str = "RL_COST") -> None:
        super().__init__(
            simulation_object=simulation_object,
            opt_method_name=name,
            config_file=None,
            verbose=False,
            set_initial_obs=False,
        )
        # Override base directory to keep RL assets under the RL folder
        self.base_dir = os.path.join("RL", name)
        os.makedirs(self.base_dir, exist_ok=True)

    def optimize(self):
        raise NotImplementedError("CostEvaluator does not implement an optimization routine.")


class DroneTrajectoryEnv(Env):
    """Environment where the agent sequentially designs intermediate waypoints."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation: Simulation,
        cost_evaluator: CostEvaluator,
        start_point: Tuple[float, float, float],
        final_target: Tuple[float, float, float],
        max_waypoints: int,
        action_bounds: Dict[str, List[float]],
        termination_distance: float,
        cost_parameters: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.simulation = simulation
        self.cost_evaluator = cost_evaluator
        self.start_point = np.array(start_point, dtype=float)
        self.final_target = np.array(final_target, dtype=float)
        self.max_waypoints = int(max_waypoints)
        self.termination_distance = float(termination_distance)
        self.cost_parameters = cost_parameters or {}

        self.low_action = np.array(action_bounds.get("low", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.high_action = np.array(action_bounds.get("high", [1000.0, 1000.0, 200.0, 20.0]), dtype=np.float32)
        if self.low_action.shape != (4,) or self.high_action.shape != (4,):
            raise ValueError("action_bounds must define 'low' and 'high' arrays with four elements each.")

        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        state_dim = self._compute_state_dimension()
        obs_low = np.full(state_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(state_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.current_waypoints: List[Dict[str, float]] = []
        self.current_step = 0
        self._needs_reset = True
        self.last_episode_data: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        del options  # unused

        self.current_waypoints = []
        self.current_step = 0
        self._needs_reset = True
        self.last_episode_data = None

        # Restore simulation state to the initial conditions
        self.simulation._clear_histories()  # type: ignore[attr-defined]
        self.simulation.drone.reset_state()
        self.simulation.drone.state['pos'] = self.start_point.copy()
        self.simulation.drone.init_state['pos'] = self.start_point.copy()
        self.simulation.current_seg_idx = 0
        self.simulation.waypoints = []

        observation = self._get_observation()
        info: Dict = {}
        return observation, info

    def step(self, action: np.ndarray):  # type: ignore[override]
        action = np.clip(action, self.low_action, self.high_action).astype(float)
        waypoint = self._build_waypoint(action)
        self.current_waypoints.append(waypoint)
        self.simulation.waypoints = [deepcopy(waypoint)]
        self.simulation.current_seg_idx = 0

        reset_state = self._needs_reset
        self._needs_reset = False

        self.simulation.startSimulation(
            stop_at_target=True,
            verbose=False,
            stop_sim_if_not_moving=True,
            use_static_target=False,
            reset_drone_state=reset_state,
        )

        costs = self._calculate_costs()
        reward = -float(costs['total_cost'])

        terminated = False
        truncated = False
        info: Dict = {
            'costs': costs,
            'waypoint': waypoint,
            'trajectory': deepcopy(self.current_waypoints),
        }

        self.current_step += 1
        drone_position = self.simulation.drone.state['pos']
        reached_goal = np.linalg.norm(drone_position - self.final_target) <= self.termination_distance
        exhausted_budget = self.current_step >= self.max_waypoints

        if reached_goal or exhausted_budget:
            if not reached_goal:
                # Force the final leg toward the endpoint
                final_wp = self._build_waypoint(np.concatenate((self.final_target, [self.high_action[-1]])))
                self.simulation.waypoints = [final_wp]
                self.simulation.current_seg_idx = 0
                self.simulation.startSimulation(
                    stop_at_target=True,
                    verbose=False,
                    stop_sim_if_not_moving=True,
                    use_static_target=False,
                    reset_drone_state=False,
                )
                self.current_waypoints.append(final_wp)
                costs = self._calculate_costs()
                reward = -float(costs['total_cost'])
                info['costs'] = costs
                info['trajectory'] = deepcopy(self.current_waypoints)
            else:
                # Ensure the final target is explicitly stored in the trajectory
                if np.linalg.norm(self.final_target - np.array([
                    info['trajectory'][-1]['x'],
                    info['trajectory'][-1]['y'],
                    info['trajectory'][-1]['z']
                ])) > 1e-6:
                    final_wp = self._build_waypoint(np.concatenate((self.final_target, [info['trajectory'][-1]['v']])))
                    self.current_waypoints.append(final_wp)
                    info['trajectory'] = deepcopy(self.current_waypoints)

            terminated = True
            self.last_episode_data = {
                'trajectory': deepcopy(self.current_waypoints),
                'total_cost': float(costs['total_cost']),
                'costs': {k: float(v) for k, v in costs.items()},
            }
            info['episode_data'] = deepcopy(self.last_episode_data)

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _calculate_costs(self) -> Dict[str, float]:
        kwargs = dict(self.cost_parameters)
        kwargs.setdefault('save_costs_in_history', False)
        result = self.cost_evaluator.calculate_costs(**kwargs)
        return {k: float(v) for k, v in result.items()}

    def _build_waypoint(self, action: np.ndarray) -> Dict[str, float]:
        return {
            'x': float(action[0]),
            'y': float(action[1]),
            'z': float(action[2]),
            'v': float(action[3]),
        }

    def _get_observation(self) -> np.ndarray:
        state = self.simulation.drone.state
        obs_components = [
            np.asarray(state.get('pos', np.zeros(3)), dtype=np.float32),
            np.asarray(state.get('vel', np.zeros(3)), dtype=np.float32),
            np.asarray(state.get('angles', np.zeros(3)), dtype=np.float32),
            np.asarray(state.get('ang_vel', np.zeros(3)), dtype=np.float32),
            np.asarray(state.get('rpm', np.zeros(4)), dtype=np.float32),
        ]
        thrust = state.get('thrust', 0.0)
        power = state.get('power', 0.0)
        thrust_val = float(np.sum(thrust)) if np.size(thrust) else float(thrust)
        power_val = float(np.sum(power)) if np.size(power) else float(power)
        obs_components.append(np.array([thrust_val, power_val], dtype=np.float32))
        steps_left = np.array([float(self.max_waypoints - self.current_step)], dtype=np.float32)
        obs_components.append(steps_left)
        return np.concatenate(obs_components, dtype=np.float32)

    def _compute_state_dimension(self) -> int:
        dummy_state = {
            'pos': np.zeros(3),
            'vel': np.zeros(3),
            'angles': np.zeros(3),
            'ang_vel': np.zeros(3),
            'rpm': np.zeros(4),
            'thrust': 0.0,
            'power': 0.0,
        }
        dummy_obs = [
            np.asarray(dummy_state['pos'], dtype=np.float32),
            np.asarray(dummy_state['vel'], dtype=np.float32),
            np.asarray(dummy_state['angles'], dtype=np.float32),
            np.asarray(dummy_state['ang_vel'], dtype=np.float32),
            np.asarray(dummy_state['rpm'], dtype=np.float32),
            np.array([float(dummy_state['thrust']), float(dummy_state['power'])], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        ]
        return int(np.sum([arr.size for arr in dummy_obs]))
