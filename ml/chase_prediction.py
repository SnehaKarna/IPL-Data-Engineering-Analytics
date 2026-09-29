"""Adapter for the existing scikit-learn chase model artifact."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from rag.settings import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "chase_model.joblib"


class ChasePredictionTool:
    """Load a fitted sklearn model and predict from an explicit feature mapping.

    Expected artifact: a fitted estimator/pipeline exposing ``predict_proba`` and
    ``feature_names_in_``. Training is available separately in ``ml.train_chase_model``.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = model_path or os.getenv("CHASE_MODEL_PATH") or DEFAULT_MODEL_PATH
        self.model_path = Path(configured)
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def required_features(self) -> list[str]:
        model = self._load_model()
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            raise RuntimeError(
                "The chase model artifact must expose feature_names_in_ so the agent can validate inputs."
            )
        return [str(name) for name in names]

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        model = self._load_model()
        names = self.required_features()
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"Prediction needs these model features: {', '.join(missing)}")
        frame = pd.DataFrame([{name: features[name] for name in names}], columns=names)
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        positive_index = next((i for i, label in enumerate(classes) if str(label).casefold() in {"1", "true", "win", "won"}), None)
        if positive_index is None:
            if len(probabilities) != 2:
                raise RuntimeError(f"Expected a binary chase model; found {len(probabilities)} class probabilities.")
            positive_index = 1
        return {
            "win_probability": float(probabilities[positive_index]),
            "model_path": str(self.model_path),
            "features": {name: features[name] for name in names},
        }

    def _load_model(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Chase model artifact not found at {self.model_path}. Add the existing fitted model there "
                "or configure CHASE_MODEL_PATH in .env."
            )
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Install the project requirements to load the chase model artifact.") from exc
        return joblib.load(self.model_path)
