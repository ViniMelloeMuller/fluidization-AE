import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from tensorflow import keras
from tensorflow.python.util.lazy_loader import KerasLazyLoader
from tqdm import tqdm

from notebooks.utils.helper import (
    folder_to_sequence,
    df_to_sequence,
    Calibrator,
    MinMaxScaler_AE,
    train_test_split_ae,
)

from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import (
    Dense,
    LSTM,
    Input,
    RepeatVector,
    TimeDistributed,
    Flatten,
    Dropout,
)

PARAMETERS = {
    "window_size": 20,
    "n": 100,
    "dr": 0.25,
    "l1": 0.0001,
    "l2": 0.0001,
}


def create_model(PARAMETERS: dict) -> keras.Model:
    n: int = PARAMETERS["n"]
    dr: float = PARAMETERS["dr"]
    l1: float = PARAMETERS["l1"]
    l2: float = PARAMETERS["l2"]

    autoencoder = Sequential(
        [
            Input(
                (PARAMETERS["window_size"], 1)
            ),  # Indica que as séries temporais são de apenas uma feature
            LSTM(
                n,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            Dropout(dr),
            LSTM(
                n // 2,
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            RepeatVector(window_size),
            LSTM(
                n // 2,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            Dropout(dr),
            LSTM(
                n,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            TimeDistributed(
                Dense(1, activation="linear", kernel_regularizer=l1_l2(l1=l1, l2=l2))
            ),
            Flatten(),
        ]
    )
    return autoencoder


if __name__ == "__main__":
    plt.style.use(["science", "ieee", "notebook"])

    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (9, 4)

    calibrator = Calibrator()
    scaler = MinMaxScaler_AE()

    window_size = PARAMETERS["window_size"]
    bigX = folder_to_sequence("VIDRO-B3", window_size=window_size)

    Xtrain, Xval = train_test_split_ae(bigX)
    scaler.fit(bigX)

    Xtrain_N, Xval_N = scaler.transform(Xtrain), scaler.transform(Xval)

    autoencoder = create_model(PARAMETERS)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/best_model.keras",
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
    ]

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    history = autoencoder.fit(
        Xtrain_N,
        Xtrain_N,
        validation_data=(Xval_N, Xval_N),
        epochs=500,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
    )

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = np.arange(1, len(loss) + 1)

    plt.yscale("log", base=10)
    plt.plot(epochs, loss, c="k", label="Treino", lw=3)
    plt.plot(epochs, val_loss, c="grey", label="Validação", lw=3)
    plt.scatter(
        np.argmin(val_loss), np.min(val_loss), label=f"Mínimo = {np.min(val_loss):.2E}"
    )
    plt.ylabel("Erro Médio Quadrático")
    plt.xlabel("Iteração de Treino")
    plt.xlim((0, len(loss)))
    plt.legend()

    plt.savefig("images/curva-de-aprendizdo.pdf", dpi=300, bbox_inches="tight")
    plt.show()
