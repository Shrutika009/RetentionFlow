import sys

import numpy as np
from scipy.special import logit


class CalibratedRetentionModel:
    """Compatibility class for the model object saved by Retentionflow.ipynb."""

    def __init__(self, base, calibrator, method):
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X):
        raw_probabilities = self.base.predict_proba(X)[:, 1]
        if self.method == "platt":
            probabilities = self.calibrator.predict_proba(logit(raw_probabilities).reshape(-1, 1))[:, 1]
        else:
            probabilities = self.calibrator.predict(raw_probabilities)
        return np.column_stack([1 - probabilities, probabilities])

    def get_feature_importance(self):
        return self.base.get_feature_importance()


def register_notebook_model_class() -> None:
    sys.modules["__main__"].CalibratedRetentionModel = CalibratedRetentionModel
