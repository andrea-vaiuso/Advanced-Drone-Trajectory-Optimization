"""Base optimizer class used by PID optimization scripts.

This class provides a minimal interface storing common configuration values.
Subclasses are expected to implement the :meth:`optimize` method and may
extend the initializer to parse configuration files or set up logging.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


class Optimizer:
    """Base class for optimization routines."""

    def __init__(
        self,
        name: str,
        config_file: str,
        parameters_file: str,
        *,
        verbose: bool = True,
        set_initial_obs: bool = True,
        simulate_wind_flag: bool = False,
        study_name: str = "",
        waypoints: Optional[list] = None,
        simulation_time: int = 150,
    ) -> None:
        self.name = name
        self.config_file = config_file
        self.parameters_file = parameters_file
        self.verbose = verbose
        self.set_initial_obs = set_initial_obs
        self.simulate_wind_flag = simulate_wind_flag
        self.study_name = study_name
        self.waypoints = waypoints
        self.simulation_time = simulation_time

        # Placeholders for attributes typically populated in a full implementation
        self.cfg: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}
        self.world: Any = None
        self.thrust_max: Any = None
        self.log_path: str = ""
        self.opt_output_path: str = ""
        self.iteration: int = 0
        self.best_cost: float = float("inf")
        self.best_costs: list[float] = []
        self.costs: list[float] = []
        self.noise_model: Any = None

    # ------------------------------------------------------------------
    def optimize(self) -> None:
        """Run the optimization routine."""
        raise NotImplementedError("Subclasses must implement the optimize method")

