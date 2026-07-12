"""
predictability_score.forecast.tensorflow_backend
=================================================

TensorFlow based forecastability backend.

This backend uses a lightweight LSTM model
to estimate practical forecastability.

Metric:
    Out-of-sample R²

The final score is:

    clip(R²,0,1) * 100

TensorFlow is an optional dependency.
"""

from __future__ import annotations


from typing import Any, Dict


import gc

import numpy as np


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score


from .base import ForecastBackend



class TensorFlowBackend(ForecastBackend):
    """
    TensorFlow LSTM Forecastability Backend.

    Parameters
    ----------
    window : int
        Historical sequence length.

    forecast_horizon : int
        Prediction horizon.

    epochs : int
        Maximum training epochs.

    batch_size : int
        Training batch size.

    verbose : bool
        TensorFlow training verbosity.
    """

    def __init__(
        self,
        window: int = 32,
        forecast_horizon: int = 1,
        epochs: int = 30,
        batch_size: int = 512,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:

        super().__init__(
            random_state=random_state,
            verbose=verbose,
        )

        self.window = window

        self.forecast_horizon = forecast_horizon

        self.epochs = epochs

        self.batch_size = batch_size



    # -------------------------------------------------

    def _import_tensorflow(self):

        try:

            import tensorflow as tf

            return tf

        except ImportError as exc:

            raise ImportError(

                "TensorFlow backend requires TensorFlow. "
                "Install it with: "
                "pip install tensorflow"

            ) from exc



    # -------------------------------------------------

    def _build_dataset(
        self,
        x: np.ndarray,
    ):

        X = []

        y = []


        limit = len(x) - self.forecast_horizon + 1


        for i in range(
            self.window,
            limit
        ):

            X.append(

                x[
                    i-self.window:i
                ]

            )

            y.append(

                x[
                    i+self.forecast_horizon-1
                ]

            )


        X = np.asarray(
            X,
            dtype=np.float32
        )


        y = np.asarray(
            y,
            dtype=np.float32
        )


        X = X.reshape(

            X.shape[0],

            X.shape[1],

            1

        )


        return X, y



    # -------------------------------------------------

    def _build_model(
        self,
        tf,
    ):

        model = tf.keras.Sequential(

            [

                tf.keras.layers.LSTM(

                    16,

                    input_shape=(

                        self.window,

                        1

                    )

                ),

                tf.keras.layers.Dense(1)

            ]

        )


        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3
            ),

            loss="mse"

        )


        return model



    # -------------------------------------------------

    def evaluate(
        self,
        series: Any,
    ) -> Dict[str, Any]:


        tf = self._import_tensorflow()


        tf.keras.utils.set_random_seed(

            self.random_state

        )


        x = np.asarray(

            series,

            dtype=np.float32

        )


        x = x[

            np.isfinite(x)

        ]


        if len(x) < self.window + 100:

            raise ValueError(

                "Series is too short for LSTM forecastability."

            )



        X, y = self._build_dataset(x)



        split = int(

            len(X) * 0.8

        )


        X_train = X[:split]

        X_test = X[split:]


        y_train = y[:split]

        y_test = y[split:]



        # -----------------------------
        # Scaling
        # -----------------------------

        scaler_x = StandardScaler()

        scaler_y = StandardScaler()



        X_train = scaler_x.fit_transform(

            X_train.reshape(-1,1)

        ).reshape(

            X_train.shape

        )


        X_test = scaler_x.transform(

            X_test.reshape(-1,1)

        ).reshape(

            X_test.shape

        )


        y_train = scaler_y.fit_transform(

            y_train.reshape(-1,1)

        ).ravel()


        y_test_scaled = scaler_y.transform(

            y_test.reshape(-1,1)

        ).ravel()



        # -----------------------------
        # Dataset pipeline
        # -----------------------------

        train_ds = (

            tf.data.Dataset

            .from_tensor_slices(

                (
                    X_train,
                    y_train
                )

            )

            .shuffle(2048)

            .batch(
                self.batch_size
            )

            .prefetch(
                tf.data.AUTOTUNE
            )

        )


        test_ds = (

            tf.data.Dataset

            .from_tensor_slices(

                (
                    X_test,
                    y_test_scaled
                )

            )

            .batch(
                self.batch_size
            )

        )



        model = self._build_model(tf)



        callbacks = [

            tf.keras.callbacks.EarlyStopping(

                monitor="val_loss",

                patience=3,

                restore_best_weights=True

            )

        ]



        history = model.fit(

            train_ds,

            validation_data=test_ds,

            epochs=self.epochs,

            verbose=1 if self.verbose else 0,

            callbacks=callbacks

        )



        pred = model.predict(

            X_test,

            verbose=0

        ).ravel()



        r2 = r2_score(

            y_test_scaled,

            pred

        )


        score = np.clip(

            r2,

            0,

            1

        ) * 100



        details = {

            "backend":
                "tensorflow",

            "model":
                "LSTM(16)",

            "r2":
                float(r2),

            "epochs_used":
                len(
                    history.history["loss"]
                ),

            "samples_train":
                len(X_train),

            "samples_test":
                len(X_test),

        }


        # آزادسازی حافظه TensorFlow

        tf.keras.backend.clear_session()

        gc.collect()



        return {

            "score":
                float(score),

            "details":
                details

        }