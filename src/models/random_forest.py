import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, config=None):
        super().__init__("Random Forest", config)

    def build(self, input_shape, output_shape):
        self.output_shape = output_shape
        params = {
            "n_estimators": self.config.get("n_estimators", 200),
            "max_depth": self.config.get("max_depth", 20),
            "min_samples_split": self.config.get("min_samples_split", 5),
            "min_samples_leaf": self.config.get("min_samples_leaf", 2),
            "n_jobs": -1,
            "random_state": 42,
        }
        if output_shape > 1:
            self.model = MultiOutputRegressor(RandomForestRegressor(**params))
        else:
            self.model = RandomForestRegressor(**params)
        self.logger.info("Built %s model with params: %s", self.name, params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        self.logger.info("Training %s on %d samples...", self.name, len(X_train))
        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start
        self.logger.info("%s training completed in %.2fs", self.name, self.training_time)

    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def get_feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "estimators_"):
            importances = np.mean(
                [est.feature_importances_ for est in self.model.estimators_], axis=0
            )
            return importances
        return None
