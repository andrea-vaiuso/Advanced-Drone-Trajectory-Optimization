"""Training utilities for SAC-based drone trajectory optimization."""

import json
import os
from math import isclose

import yaml
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC

from Drone.Simulation import Simulation
from Optimizations.optimizer import Optimizer
from Worlds.World import World
from RL.callbacks import RLEpisodeLogger
from RL.environment import CostEvaluator, DroneTrajectoryEnv
from main import (
    create_initial_state,
    create_quadcopter_controller,
    create_quadcopter_model,
    get_max_thrust_from_rotor_model,
    load_dnn_noise_model,
    load_parameters,
    load_pid_gains,
) 

class BaseRLTrainer(Optimizer):
    """Bridge Stable-Baselines3 trainers with the metaheuristic utilities."""

    def __init__(self, config_file: str, verbose: bool = True) -> None:
        """Load the RL configuration and instantiate the shared simulation."""
        with open(config_file, "r", encoding="utf-8") as stream:
            rl_cfg = yaml.safe_load(stream)

        self.rl_config = rl_cfg
        self.simulation_parameters_path = rl_cfg["simulation_parameters_file"]
        self.simulation_parameters = load_parameters(self.simulation_parameters_path)
        self.start_point = tuple(self.simulation_parameters["start_point"])  # type: ignore[assignment]
        self.end_point = tuple(self.simulation_parameters["end_point"])  # type: ignore[assignment]

        simulation = self._create_simulation()

        opt_name = rl_cfg.get("algorithm_name", "SAC")
        super().__init__(
            simulation_object=simulation,
            opt_method_name=opt_name,
            config_file=config_file,
            verbose=verbose,
            set_initial_obs=rl_cfg.get("set_initial_obs", True),
        )
        self.base_dir = os.path.join("RL", opt_name)
        os.makedirs(self.base_dir, exist_ok=True)

        self.last_time = 0.0

        self.cost_parameters = rl_cfg.get("cost_parameters", {})
        self.best_cost = float("inf")
        self.best_trajectory = []
        self.model = None

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    
    def optimize(self):  # type: ignore[override]
        """Train the agent and return the best trajectory found."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _create_simulation(self) -> Simulation:
        """Instantiate a fresh simulation using the YAML configuration."""
        params = self.simulation_parameters
        init_state = create_initial_state(*self.start_point)
        pid_gains = load_pid_gains(params)
        thrust_max = get_max_thrust_from_rotor_model(params)
        controller = create_quadcopter_controller(init_state, pid_gains, thrust_max, params)
        drone = create_quadcopter_model(init_state, controller, params)
        world_path = params["world_data_path"]
        if os.path.exists(world_path):
            world = World.load_world(world_path)
        else:
            grid_size = int(params.get("grid_size", 10))
            max_world_size = int(params.get("max_world_size", 1000))
            world = World(grid_size=grid_size, max_world_size=max_world_size)
        noise_model = load_dnn_noise_model(params)
        simulation = Simulation(
            drone=drone,
            world=world,
            waypoints=[],
            dt=float(params["dt"]),
            max_simulation_time=float(params["simulation_time"]),
            frame_skip=int(params["frame_skip"]),
            target_reached_threshold=float(params["threshold"]),
            target_shift_threshold_distance=float(params["target_shift_threshold_distance"]),
            noise_model=noise_model,
            noise_annoyance_radius=int(params["noise_annoyance_radius"]),
            generate_sound_emission_map=False,
            compute_psychoacoustics=False,
        )
        return simulation

    def _build_env_factory(self):
        """Return a callable that constructs isolated environment instances."""
        max_waypoints = int(self.rl_config.get("max_waypoints", 10))
        termination_distance = float(self.rl_config.get("termination_distance", 5.0))
        action_bounds = self.rl_config.get("action_bounds", {})

        def _factory():
            simulation = self._create_simulation()
            cost_evaluator = CostEvaluator(simulation, name=f"{self.opt_method_name}_COST", mkdirs=False)
            env = DroneTrajectoryEnv(
                simulation=simulation,
                cost_evaluator=cost_evaluator,
                start_point=self.start_point,
                final_target=self.end_point,
                max_waypoints=max_waypoints,
                action_bounds=action_bounds,
                termination_distance=termination_distance,
                cost_parameters=self.cost_parameters,
            )
            return env

        return _factory

    def _run_zero_waypoint_episode(self, env_factory) -> None:
        """Execute the baseline straight-line episode prior to training."""
        env = env_factory()
        try:
            _, _, _, _, info = env.run_zero_waypoint_episode()
            episode_data = info.get("episode_data") if info else None
            if episode_data:
                self.record_episode(episode_data)
        finally:
            env.close()
    
    def _run_specific_waypoint_episode(self, env_factory, waypoints) -> None:
        """Execute an episode with the provided waypoints."""
        env = env_factory()
        try:
            _, _, _, _, info = env.run_specific_waypoints_episode(waypoints)
            episode_data = info.get("episode_data") if info else None
            if episode_data:
                self.record_episode(episode_data)
        finally:
            env.close()

    def record_episode(self, episode_data) -> None:
        """Store an episode summary and update best-trajectory bookkeeping."""
        total_cost = float(episode_data["total_cost"])
        trajectory = self._convert_waypoints(episode_data["trajectory"])
        absolute_trajectory = self._finalize_trajectory(trajectory)
        self.costs_history.append(total_cost)
        self.costs_dict_history.append(episode_data["costs"])
        self.log_step(absolute_trajectory)
        if total_cost < self.best_cost:
            self.best_cost = total_cost
            self.best_trajectory = absolute_trajectory

    @staticmethod
    def _convert_waypoints(raw_waypoints):
        """Cast waypoint dictionaries to floats for downstream consumers."""
        return [
            {
                "x": float(wp["x"]),
                "y": float(wp["y"]),
                "z": float(wp["z"]),
                "v": float(wp["v"]),
            }
            for wp in raw_waypoints
        ]


    def _finalize_trajectory(self, waypoints):
        """Ensure the terminal target waypoint is present in the trajectory."""
        sanitized = [
            {
                "x": float(wp["x"]),
                "y": float(wp["y"]),
                "z": float(wp["z"]),
                "v": float(wp["v"]),
            }
            for wp in waypoints
        ]

        if not sanitized:
            return sanitized

        last = sanitized[-1]
        target_x, target_y, target_z = self.end_point
        if not (
            isclose(last["x"], target_x, abs_tol=1e-6)
            and isclose(last["y"], target_y, abs_tol=1e-6)
            and isclose(last["z"], target_z, abs_tol=1e-6)
        ):
            sanitized.append(
                {
                    "x": float(target_x),
                    "y": float(target_y),
                    "z": float(target_z),
                    "v": float(last["v"]),
                }
            )

        return sanitized

    def save_results_in_file(self, best_params) -> None:  # type: ignore[override]
        """Persist RL optimisation results to disk using absolute waypoints."""

        serialized_best = self._finalize_trajectory(best_params) if best_params else []
        results = {
            "Optimization Algorithm": self.opt_method_name,
            "Best Parameters Found": serialized_best,
            "Best Cost": float(self.best_cost) if self.costs_history else None,
            "Total time (s)": self.end_time,
            "Number of iterations": len(self.costs_history),
            "Avg step time (s)": self.end_time / len(self.costs_history) if self.costs_history else None,
            "Optimization Parameters": self.cfg,
        }
        results_path = os.path.join(self.study_dir, "optimization_results.json")
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"{self.get_alg_prefix()} Results saved to {results_path}")


class SACTrajectoryTrainer(BaseRLTrainer):
    """Trainer that leverages Stable-Baselines3 SAC."""
    def __init__(self, config_file, verbose = True, specific_waypoints = None, run_zero_waypoint_episode = True) -> None:
        super().__init__(config_file, verbose)
        self.specific_waypoints = specific_waypoints
        self.run_zero_waypoint_episode = run_zero_waypoint_episode


    def optimize(self):  # type: ignore[override]
        """Train the SAC agent and return the best recorded trajectory."""
        env_factory = self._build_env_factory()
        if self.run_zero_waypoint_episode:
            self._run_zero_waypoint_episode(env_factory)
        if self.specific_waypoints is not None:
            self._run_specific_waypoint_episode(env_factory, self.specific_waypoints)

        train_env = DummyVecEnv([env_factory])

        sac_kwargs = self._build_sac_kwargs(train_env)
        model = SAC(**sac_kwargs)
        self.model = model

        callback = RLEpisodeLogger(self)
        total_timesteps = int(self.rl_config.get("total_timesteps", 10_000))
        log_interval = self.rl_config.get("log_interval", 1)

        try:
            model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval)
        finally:
            train_env.close()

        if self.study_dir:
            model_path = os.path.join(self.study_dir, "sac_model")
            model.save(model_path)
            print(f"{self.get_alg_prefix()} Model saved to {model_path}")

        eval_episodes = int(self.rl_config.get("evaluation_episodes", 3))
        if eval_episodes > 0:
            self._evaluate_policy(model, env_factory, eval_episodes)

        return self.best_trajectory

    def _evaluate_policy(self, model: SAC, env_factory, episodes: int) -> None:
        """Evaluate the deterministic policy for the requested number of episodes."""
        for _ in range(episodes):
            env = env_factory()
            obs, _ = env.reset()
            done = False
            truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
            episode_data = info.get("episode_data")
            if episode_data:
                self.record_episode(episode_data)
            env.close()

    def _build_sac_kwargs(self, env: DummyVecEnv):
        """Translate the YAML configuration into SAC constructor arguments."""
        policy = self.rl_config.get("policy", "MlpPolicy")
        kwargs = {
            "policy": policy,
            "env": env,
            "learning_rate": self.rl_config.get("learning_rate", 3e-4),
            "buffer_size": int(self.rl_config.get("buffer_size", 100_000)),
            "batch_size": int(self.rl_config.get("batch_size", 256)),
            "tau": self.rl_config.get("tau", 0.005),
            "gamma": self.rl_config.get("gamma", 0.99),
            "train_freq": self.rl_config.get("train_freq", 1),
            "gradient_steps": self.rl_config.get("gradient_steps", 1),
            "ent_coef": self.rl_config.get("ent_coef", "auto"),
            "target_update_interval": int(self.rl_config.get("target_update_interval", 1)),
            "learning_starts": int(self.rl_config.get("learning_starts", 1_000)),
            "device": self.rl_config.get("device", "auto"),
        }
        if self.rl_config.get("policy_kwargs"):
            kwargs["policy_kwargs"] = self.rl_config["policy_kwargs"]
        return kwargs
