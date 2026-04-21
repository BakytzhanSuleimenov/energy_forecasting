import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np

from src.models.base import BaseModel


class DNNModel(BaseModel):
    def __init__(self, config=None):
        super().__init__("DNN", config)
        self.history = None

    def build(self, input_shape, output_shape):
        from keras import Input, layers, models, optimizers

        self.output_shape = output_shape
        hidden_layers = self.config.get("hidden_layers", [128, 64, 32])
        dropout_rate = self.config.get("dropout_rate", 0.2)
        learning_rate = self.config.get("learning_rate", 0.001)

        inputs = Input(shape=(input_shape,))
        x = inputs
        for units in hidden_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(output_shape)(x)

        self.model = models.Model(inputs, outputs)
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        self.logger.info(
            "Built %s model: layers=%s, dropout=%.2f, lr=%.4f",
            self.name, hidden_layers, dropout_rate, learning_rate,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)

        epochs = self.config.get("epochs", 100)
        batch_size = self.config.get("batch_size", 64)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.logger.info("Training %s on %d samples...", self.name, len(X_train))
        start = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )
        self.training_time = time.time() - start
        self.logger.info("%s training completed in %.2fs", self.name, self.training_time)

    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X, verbose=0)

    def get_training_history(self):
        if self.history is None:
            return None
        return self.history.history
