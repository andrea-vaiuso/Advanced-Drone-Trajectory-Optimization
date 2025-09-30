"""Gymnasium environment for drone trajectory optimization with SAC."""

import os
from copy import deepcopy

import numpy as np
from gymnasium import Env, spaces

from Drone.Simulation import Simulation
from Optimizations.optimizer import MetaHeuristicOptimizer


class CostEvaluator(MetaHeuristicOptimizer):
    """Expose the metaheuristic cost evaluator without running an optimiser."""

    def __init__(self, simulation_object, name: str = "RL_COST") -> None:
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
    """Gymnasium environment that lets an agent author waypoint perturbations.

    The environment interprets actions as offsets around reference points
    placed on the straight segment between the start and target locations.  A
    final action component optionally terminates waypoint generation early so
    the agent can choose between trajectories with different waypoint counts.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation: Simulation,
        cost_evaluator: CostEvaluator,
        start_point: np.ndarray,
        final_target: np.ndarray,
        max_waypoints: int,
        action_bounds: dict,
        termination_distance: float,
        cost_parameters: dict = None,
    ):
        super().__init__()
        self.simulation = simulation
        self.cost_evaluator = cost_evaluator
        self.start_point = np.array(start_point, dtype=float)
        self.final_target = np.array(final_target, dtype=float)
        self.max_waypoints = int(max_waypoints)
        self.termination_distance = float(termination_distance)
        self.cost_parameters = cost_parameters or {}

        raw_low = np.array(action_bounds.get("low", [-250.0, -250.0, -250.0, 2.0]), dtype=float)
        raw_high = np.array(action_bounds.get("high", [250.0, 250.0, 250.0, 20.0]), dtype=float)

        if raw_low.shape not in {(4,), (5,)} or raw_high.shape != raw_low.shape:
            raise ValueError(
                "action_bounds must define 'low' and 'high' arrays with four or five elements each."
            )

        if raw_low.shape == (4,):
            raw_low = np.concatenate([raw_low, [0.0]])
            raw_high = np.concatenate([raw_high, [1.0]])

        self.world_min_bounds, self.world_max_bounds = self._infer_world_bounds()
        (
            self.low_action,
            self.high_action,
            self.per_waypoint_low,
            self.per_waypoint_high,
        ) = self._initialize_action_bounds(raw_low, raw_high)

        self.speed_index = 3
        self.stop_action_index = 4 if self.low_action.size > 4 else None
        self.stop_threshold = 0.5

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

        self.current_waypoints = []
        self.current_step = 0
        self._needs_reset = True
        self.last_episode_data = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        """Reset the simulator state and return the initial observation."""
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
        info = {}
        return observation, info

    def step(self, action: np.ndarray):  # type: ignore[override]
        """Apply an action and advance the environment by one waypoint step.

        Args:
            action: A numpy array of shape (4,) or (5,) representing the action to be taken.
        Returns:
            observation: The next observation of the environment.
            reward: The reward obtained from taking the action.
            terminated: A boolean indicating if the episode has ended.
            truncated: A boolean indicating if the episode was truncated.
            info: A dictionary containing additional information about the step.
        """
        action = np.clip(action, self.low_action, self.high_action).astype(float)

        if self._should_request_stop(action):
            return self._finalize_episode(
                reached_goal=False,
                forced_speed=float(action[self.speed_index]),
                stopped_by_agent=True,
                last_waypoint=None,
            )

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
        info = {
            'costs': costs,
            'waypoint': waypoint,
            'trajectory': deepcopy(self.current_waypoints),
            'stopped_by_agent': False,
        }

        self.current_step += 1
        drone_position = self.simulation.drone.state['pos']
        reached_goal = np.linalg.norm(drone_position - self.final_target) <= self.termination_distance
        exhausted_budget = self.current_step >= self.max_waypoints

        if reached_goal or exhausted_budget:
            return self._finalize_episode(
                reached_goal=reached_goal,
                forced_speed=float(action[self.speed_index]),
                stopped_by_agent=False,
                last_waypoint=waypoint,
                cached_costs=costs if reached_goal else None,
            )

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def run_zero_waypoint_episode(self):
        """Simulate the straight-line baseline without inserting waypoints."""
        self.reset()
        default_speed = self._default_terminal_speed()
        return self._finalize_episode(
            reached_goal=False,
            forced_speed=default_speed,
            stopped_by_agent=True,
            last_waypoint=None,
        )

    def _calculate_costs(self):
        """Return the current cost metrics computed by the shared evaluator."""
        kwargs = dict(self.cost_parameters)
        kwargs.setdefault('save_costs_in_history', False)
        result = self.cost_evaluator.calculate_costs(**kwargs)
        return {k: float(v) for k, v in result.items()}

    def _build_waypoint(self, action: np.ndarray, index: int):
        """Convert an action into an absolute waypoint for the given index."""
        reference = self._get_reference_point(index)
        offset = action[:3]
        position = reference + offset
        return self._build_absolute_waypoint(position, float(action[self.speed_index]))

    def _should_request_stop(self, action: np.ndarray) -> bool:
        """Return ``True`` when the action requests early termination."""

        if self.stop_action_index is None:
            return False
        stop_signal = float(action[self.stop_action_index])
        return stop_signal <= self.stop_threshold

    def _finalize_episode(
        self,
        reached_goal,
        forced_speed,
        stopped_by_agent,
        last_waypoint,
        cached_costs=None,
    ):
        """Assemble the terminal transition and associated bookkeeping."""
        if reached_goal:
            final_wp = self._ensure_final_target_present(
                speed=(last_waypoint['v'] if last_waypoint else forced_speed),
                run_simulation=False,
            )
            costs = cached_costs if cached_costs is not None else self._calculate_costs()
            reward = -float(costs['total_cost'])
        else:
            final_wp = self._ensure_final_target_present(speed=forced_speed, run_simulation=True)
            costs = self._calculate_costs()
            reward = -float(costs['total_cost'])

        info = {
            'costs': costs,
            'waypoint': last_waypoint if last_waypoint is not None else final_wp,
            'trajectory': deepcopy(self.current_waypoints),
            'stopped_by_agent': stopped_by_agent,
        }

        self.last_episode_data = {
            'trajectory': deepcopy(self.current_waypoints),
            'total_cost': float(costs['total_cost']),
            'costs': {k: float(v) for k, v in costs.items()},
        }
        info['episode_data'] = deepcopy(self.last_episode_data)

        observation = self._get_observation()
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info

    def _ensure_final_target_present(self, speed, run_simulation):
        """Append the terminal waypoint when missing and optionally simulate it."""
        last_wp = self.current_waypoints[-1] if self.current_waypoints else None
        if last_wp is not None:
            last_pos = np.array([last_wp['x'], last_wp['y'], last_wp['z']], dtype=float)
            if np.linalg.norm(last_pos - self.final_target) <= 1e-6:
                return last_wp

        base_speed = speed if speed is not None else (
            last_wp['v'] if last_wp is not None else self._default_terminal_speed()
        )
        final_wp = self._build_absolute_waypoint(self.final_target, float(base_speed))

        if run_simulation:
            reset_state = self._needs_reset
            self.simulation.waypoints = [deepcopy(final_wp)]
            self.simulation.current_seg_idx = 0
            self.simulation.startSimulation(
                stop_at_target=True,
                verbose=False,
                stop_sim_if_not_moving=True,
                use_static_target=False,
                reset_drone_state=reset_state,
            )
            self._needs_reset = False

        self.current_waypoints.append(final_wp)
        return final_wp

    def _default_terminal_speed(self):
        """Return the midpoint between the minimum and maximum speed bounds."""
        low = float(self.low_action[self.speed_index])
        high = float(self.high_action[self.speed_index])
        midpoint = 0.5 * (low + high)
        return float(np.clip(midpoint, low, high))

    def _get_reference_point(self, index):
        """Return the neutral waypoint corresponding to the given index."""
        if self.reference_points.size == 0:
            return self.final_target.astype(float)
        clipped_index = min(max(index, 0), len(self.reference_points) - 1)
        return self.reference_points[clipped_index]

    @staticmethod
    def _build_absolute_waypoint(position: np.ndarray, speed: float):
        """Create a waypoint dictionary from position and speed arrays."""
        return {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'v': float(speed),
        }

    def _get_observation(self):
        """Assemble the observation vector from the drone telemetry."""
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

    def _compute_state_dimension(self):
        """Return the size of the flattened observation vector."""
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


    def _infer_world_bounds(self):
        """Infer the spatial bounds of the environment from the simulation."""
        world = getattr(self.simulation, "world", None)
        world_min = np.zeros(3, dtype=float)
        world_max_value = None
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

    def _initialize_action_bounds(self, raw_low: np.ndarray, raw_high: np.ndarray):
        """Compute global and per-waypoint action bounds for the SAC policy."""
        if self.max_waypoints <= 0:
            low_action = raw_low.astype(np.float32)
            high_action = raw_high.astype(np.float32)
            empty = np.zeros((0, raw_low.size), dtype=float)
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
        base_low_core = np.broadcast_to(raw_low[:4], (self.max_waypoints, 4))
        base_high_core = np.broadcast_to(raw_high[:4], (self.max_waypoints, 4))

        per_waypoint_low_core = np.maximum(world_low, base_low_core)
        per_waypoint_high_core = np.minimum(world_high, base_high_core)

        if raw_low.size > 4:
            gating_low = np.broadcast_to(raw_low[4:], (self.max_waypoints, raw_low.size - 4))
            gating_high = np.broadcast_to(raw_high[4:], (self.max_waypoints, raw_low.size - 4))
            per_waypoint_low = np.concatenate((per_waypoint_low_core, gating_low), axis=1)
            per_waypoint_high = np.concatenate((per_waypoint_high_core, gating_high), axis=1)
        else:
            per_waypoint_low = per_waypoint_low_core
            per_waypoint_high = per_waypoint_high_core

        if np.any(per_waypoint_low > per_waypoint_high):
            raise ValueError("Inconsistent action bounds: some waypoint intervals are invalid.")

        low_action = np.max(per_waypoint_low, axis=0).astype(np.float32)
        high_action = np.min(per_waypoint_high, axis=0).astype(np.float32)

        if np.any(low_action > high_action):
            raise ValueError(
                "Unable to derive consistent global action bounds from per-waypoint limits."
            )

        return low_action, high_action, per_waypoint_low, per_waypoint_high
