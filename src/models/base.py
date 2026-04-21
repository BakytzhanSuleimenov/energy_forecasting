import logging
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.training_time = 0
        self.logger = logging.getLogger("energy_forecasting")

    @abstractmethod
    def build(self, input_shape, output_shape):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_training_time(self):
        return self.training_time
