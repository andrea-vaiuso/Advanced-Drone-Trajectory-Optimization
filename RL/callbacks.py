"""Custom callbacks used during RL training."""

from stable_baselines3.common.callbacks import BaseCallback



class RLEpisodeLogger(BaseCallback):
    """Forward environment episode summaries to the :class:`BaseRLTrainer`."""

    def __init__(self, trainer, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.trainer = trainer

    def _on_step(self) -> bool:
        """Capture episode data produced by :class:`DroneTrajectoryEnv`."""
        infos = self.locals.get("infos", [])  # type: ignore[assignment]
        for info in infos:
            data = info.get("episode_data")
            if data:
                self.trainer.record_episode(data)
        return True
