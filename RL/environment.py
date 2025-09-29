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

        raw_low = np.array(action_bounds.get("low", [-100.0, -100.0, -100.0, 0.0]), dtype=float)
        raw_high = np.array(action_bounds.get("high", [100.0, 100.0, 100.0, 20.0]), dtype=float)
        if raw_low.shape != (4,) or raw_high.shape != (4,):
            raise ValueError("action_bounds must define 'low' and 'high' arrays with four elements each.")

        self.world_min_bounds, self.world_max_bounds = self._infer_world_bounds()
        (
            self.low_action,
            self.high_action,
            self.per_waypoint_low,
            self.per_waypoint_high,
        ) = self._initialize_action_bounds(raw_low, raw_high)

        self.action_space = spaces.Box(low=self.low_action, high=self.high_action, dtype=np.float32)

        # Reference internal waypoints on the direct A→B segment.
        self.reference_points = np.asarray(
            MetaHeuristicOptimizer.linspace_internal_points(
                self.start_point,
                self.final_target,
                self.max_waypoints,
            ),
            dtype=float,
        )

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
        action = self._clip_action_for_index(action, self.current_step)
        waypoint = self._build_waypoint(action, self.current_step)
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
                final_wp = self._build_absolute_waypoint(self.final_target, float(self.high_action[-1]))
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
                    final_wp = self._build_absolute_waypoint(
                        self.final_target,
                        float(info['trajectory'][-1]['v']),
                    )
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

    def _build_waypoint(self, action: np.ndarray, index: int) -> Dict[str, float]:
        reference = self._get_reference_point(index)
        offset = action[:3]
        position = reference + offset
        position = np.clip(position, self.world_min_bounds, self.world_max_bounds)
        return self._build_absolute_waypoint(position, float(action[3]))

    def _get_reference_point(self, index: int) -> np.ndarray:
        if self.reference_points.size == 0:
            return self.final_target.astype(float)
        clipped_index = min(max(index, 0), len(self.reference_points) - 1)
        return self.reference_points[clipped_index]

    @staticmethod
    def _build_absolute_waypoint(position: np.ndarray, speed: float) -> Dict[str, float]:
        return {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'v': float(speed),
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

    def _infer_world_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        world = getattr(self.simulation, "world", None)
        world_min = np.zeros(3, dtype=float)
        world_max_value: Optional[float] = None
        if world is not None:
            max_world_size = getattr(world, "max_world_size", None)
            if max_world_size is not None:
                try:
                    world_max_value = float(max_world_size)
                except (TypeError, ValueError):
                    world_max_value = None

        if world_max_value is None or world_max_value <= 0:
            coords = np.vstack([self.start_point, self.final_target]) if self.start_point.size else np.zeros((0, 3))
            fallback = float(np.max(coords)) if coords.size else 0.0
            world_max_value = max(fallback, 1.0)

        world_max = np.full(3, world_max_value, dtype=float)
        return world_min, world_max

    def _initialize_action_bounds(
        self, raw_low: np.ndarray, raw_high: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.max_waypoints <= 0:
            low_action = raw_low.astype(np.float32)
            high_action = raw_high.astype(np.float32)
            empty = np.zeros((0, 4), dtype=float)
            return low_action, high_action, empty, empty

        offset_candidates = np.concatenate((raw_low[:3], raw_high[:3]))
        finite_candidates = np.abs(offset_candidates[np.isfinite(offset_candidates)])
        max_offset = float(np.max(finite_candidates)) if finite_candidates.size else None
        if max_offset is not None and max_offset <= 0:
            max_offset = None

        world_low_flat, world_high_flat = MetaHeuristicOptimizer.build_particle_bounds(
            self.start_point,
            self.final_target,
            self.max_waypoints,
            max_perturbation_offset=max_offset,
            vmax=float(raw_high[3]),
            world_min=float(self.world_min_bounds[0]),
            world_max=float(self.world_max_bounds[0]),
            v_min=float(raw_low[3]),
        )

        world_low = world_low_flat.reshape(self.max_waypoints, 4)
        world_high = world_high_flat.reshape(self.max_waypoints, 4)
        base_low = np.broadcast_to(raw_low, (self.max_waypoints, 4))
        base_high = np.broadcast_to(raw_high, (self.max_waypoints, 4))

        per_waypoint_low = np.maximum(world_low, base_low)
        per_waypoint_high = np.minimum(world_high, base_high)

        if np.any(per_waypoint_low > per_waypoint_high):
            raise ValueError("Inconsistent action bounds: some waypoint intervals are invalid.")

        low_action = per_waypoint_low.min(axis=0).astype(np.float32)
        high_action = per_waypoint_high.max(axis=0).astype(np.float32)

        return low_action, high_action, per_waypoint_low, per_waypoint_high

    def _clip_action_for_index(self, action: np.ndarray, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.per_waypoint_low):
            return action
        clipped = np.clip(action, self.per_waypoint_low[index], self.per_waypoint_high[index])
        return clipped
