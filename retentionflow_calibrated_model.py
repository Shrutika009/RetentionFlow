import numpy as np


class CalibratedProbabilityModel:
    """Wrap a fitted classifier with a one-dimensional probability calibrator."""

    def __init__(self, base_model, calibrator, eps: float = 1e-6):
        self.base_model = base_model
        self.calibrator = calibrator
        self.eps = eps
        self.classes_ = np.array([0, 1])

    def _logit_probability(self, frame):
        raw_probability = self.base_model.predict_proba(frame)[:, 1]
        clipped = np.clip(raw_probability, self.eps, 1 - self.eps)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict_proba(self, frame):
        calibrated = self.calibrator.predict_proba(self._logit_probability(frame))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])
