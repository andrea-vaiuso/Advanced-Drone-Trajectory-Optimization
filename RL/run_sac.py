"""Entry point for launching SAC training."""
from __future__ import annotations

import argparse
import os

from RL.trainer import SACTrajectoryTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SAC agent for drone trajectory optimization.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("Settings", "rl_parameters.yaml"),
        help="Path to the RL configuration file.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Optional label appended to the results folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = SACTrajectoryTrainer(config_file=args.config)
    if args.study_name:
        trainer.study_name = args.study_name
    trainer.start_optimization()


if __name__ == "__main__":
    main()
