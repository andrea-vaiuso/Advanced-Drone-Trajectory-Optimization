"""Launch multiple PID optimization algorithms concurrently.

This script starts all available optimization algorithms using the
same configuration parameters. Each optimizer runs in its own
process so that they execute concurrently.

If the operating system supports separate terminal windows (e.g. on
Windows using the ``CREATE_NEW_CONSOLE`` flag), the optimizers can be
launched in individual command prompts. Otherwise they run as
background processes within the same terminal.
"""

from multiprocessing import Process
from typing import Type, List

from Optimizations.pid_optimization_ga import GAOptimizer
from Optimizations.pid_optimization_gwo import GWOOptimizer
from Optimizations.pid_optimization_pso import PSOOptimizer
from Optimizations.pid_optimization_bayopt import BayesianOptimizer
from Optimizations.optimizer import Optimizer

# List of optimizer classes to run
OPTIMIZER_CLASSES: List[Type[Optimizer]] = [
    GAOptimizer,
    PSOOptimizer,
    GWOOptimizer,
    BayesianOptimizer,
]


def run_optimizer(opt_class: Type[Optimizer], study_name: str) -> None:
    """Instantiate and execute the optimization algorithm."""
    optimizer = opt_class(
        verbose=True,
        set_initial_obs=True,
        simulate_wind_flag=False,
        study_name=study_name
    )
    optimizer.optimize()


def main() -> None:
    """Start all optimizers in separate processes."""

    study_name = "no_wind"


    processes = [
        Process(target=run_optimizer, args=(opt_class, study_name))
        for opt_class in OPTIMIZER_CLASSES
    ]

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()