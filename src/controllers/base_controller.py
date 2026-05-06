# -*- coding: utf-8 -*-
"""Shared controller interface helpers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


def obs_hour_normalized(obs: np.ndarray) -> float:
    """
    Return normalized hour-of-day from the 31D environment observation.

    The current observation design stores day progress at `obs[10]` in [-1, 1].
    This helper maps it back to [0, 1].
    """
    if len(obs) > 10:
        return float(np.clip((float(obs[10]) + 1.0) * 0.5, 0.0, 1.0))
    return 0.5


class BaseController(ABC):
    """Base class for rule-based, RL, and MPC controllers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Predict the next normalized action from the current observation."""

    def reset(self):
        """Reset internal controller state if needed."""

    def get_config(self) -> Dict[str, Any]:
        """Return a shallow copy of the controller config."""
        return self.config.copy()
