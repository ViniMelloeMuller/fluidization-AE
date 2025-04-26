from operator import call
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from tensorflow import keras
from tensorflow.python.util.lazy_loader import KerasLazyLoader
from tqdm import tqdm
import optuna
from optuna.trial import Trial

from notebooks.utils.helper import (
    folder_to_sequence,
    MinMaxScaler_AE,
)

from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import (
    Dense,
    LSTM,
    Input,
    TimeDistributed,
    Flatten,
    Dropout,
)

PARAMETERS = {
    "window_size": 20,
    "n": 40,
    "dr": 0.25,
    "l1": 0.0001,
    "l2": 0.0001,
}


def create_model(
    n_units: int, dr: float, l1: float, l2: float, window_size: int = 20
) -> keras.Model:
    """Creates a Sequential LSTM model for time series prediction."""
    seq2seq = Sequential(
        [
            Input(
                (window_size, 1)
            ),  # Indica que as séries temporais são de apenas uma feature
            LSTM(
                n_units,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            Dropout(dr),
            LSTM(
                n_units,
                return_sequences=True,
                kernel_regularizer=l1_l2(l1=l1, l2=l2),
            ),
            TimeDistributed(
                Dense(1, activation="linear", kernel_regularizer=l1_l2(l1=l1, l2=l2))
            ),
            Flatten(),
        ]
    )
    return seq2seq


def objective(trial: Trial, x_train, y_train, x_val, y_val) -> float:
    # Sugere os hiperparametros
    n_units = trial.suggest_int("n_units", 16, 128, step=16)  # Unidades LSTM
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)  # Dropout
    l1_reg = trial.suggest_float("l1", 1e-6, 1e-3, log=True)  # L1
    l2_reg = trial.suggest_float("l2", 1e-6, 1e-3, log=True)  # L2

    best_model_path = f"../models/model_{trial.number}.keras"
    trial.set_user_attr("model_path", best_model_path)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=15, min_lr=1e-5
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
        optuna.integration.KerasPruningCallback(trial, "val_loss"),
    ]

    seq2seq = create_model(n_units, dropout_rate, l1_reg, l2_reg)

    seq2seq.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    history = seq2seq.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=500,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss = seq2seq.evaluate(Xval_N, Yval_N, verbose=0)

    return val_loss


if __name__ == "__main__":
    np.random.seed(50)  # 50 anos da faculdade de engenharia química FEQ UNICAMP :)
    plt.style.use(["science", "ieee", "notebook"])

    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (9, 4)

    scaler = MinMaxScaler_AE()

    window_size = PARAMETERS["window_size"]
    X, Y = folder_to_sequence("VIDRO-B3", window_size=window_size)

    n_samples = X.shape[0]
    idx = np.random.permutation(n_samples)
    test_size = int(n_samples * 0.25)
    Xtrain, Xval = X[idx[test_size:]], X[idx[:test_size]]
    Ytrain, Yval = Y[idx[test_size:]], Y[idx[:test_size]]
    print(f"Dados divididos: Treino={Xtrain.shape[0]}, Validação={Xval.shape[0]}")

    scalerX = MinMaxScaler_AE()
    scalerY = MinMaxScaler_AE()

    scalerX.fit(X)
    scalerY.fit(Y)

    Xtrain_N, Xval_N = scalerX.transform(Xtrain), scalerX.transform(Xval)
    Ytrain_N, Yval_N = scalerY.transform(Ytrain), scalerY.transform(Yval)

    study_name = "LSTM-v2"
    storage_path = "../models/optuna_study.db"
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    storage_name = f"sqlite:///{storage_path}"

    print(f"Armazenando o estudo em {storage_name}")
    print(f"Nome do estudo: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    try:
        study.optimize(
            lambda trial: objective(trial, Xtrain_N, Ytrain_N, Xval_N, Yval_N),
            n_trials=10,
            n_jobs=1,
            timeout=600,
        )
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")

    print("Melhores hiperparâmetros encontrados:")

    if len(study.trials) > 0 and study.best_trial:
        best_trial = study.best_trial
        print(f"  Número do melhor trial: {best_trial.number}")
        print(f"  Valor da função objetivo: {best_trial.value}")
        print(f"  Hiperparâmetros: {best_trial.params}")
