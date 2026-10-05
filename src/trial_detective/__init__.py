"""Statistical detection of fabricated data in multi-site clinical trials."""

from .scoring import evaluate, run_all
from .simulate import FRAUD_TYPES, TrialConfig, simulate_trial

__all__ = ["FRAUD_TYPES", "TrialConfig", "evaluate", "run_all", "simulate_trial"]
__version__ = "0.1.0"
