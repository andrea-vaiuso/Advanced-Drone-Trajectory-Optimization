"""Custom callbacks used during RL training."""
from __future__ import annotations

from typing import Any, Dict, List

from stable_baselines3.common.callbacks import BaseCallback


class RLEpisodeLogger(BaseCallback):
    """Callback that forwards episode summaries to the trainer."""

    def __init__(self, trainer: "BaseRLTrainer", verbose: int = 0) -> None:
        super().__init__(verbose)
        self.trainer = trainer

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])  # type: ignore[assignment]
        for info in infos:
            data = info.get("episode_data")
            if data:
                self.trainer.record_episode(data)
        return True
