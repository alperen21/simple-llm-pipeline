"""Pydantic contracts for YAML configs and runtime records.

Training, sweep, selection, and deployment configs reject unknown keys.
Dataset schemas validate tool-calling conversations before SFT. RAG schemas
describe retrieved passages. Catalog schemas are reserved for persisted books.
"""

from . import deployment, experiments, training
from .deployment import DeploymentConfig
from .training import TrainingConfig

__all__ = [
    "DeploymentConfig",
    "TrainingConfig",
    "deployment",
    "experiments",
    "training",
]
