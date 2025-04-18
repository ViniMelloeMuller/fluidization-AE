import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def folder_to_sequence(
    folder_path: str, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Função usada para converter os dados .csv de uma pasta
    para o formato aceito pelo modelo FT101 -> PT105

    Parameters
    ----------
    folder_path: str
        Caminho (pasta) em que os dados estão localizados.

    window_size: int
        Tamanho da janela de dados que o modelo receberá.

    Returns
    -------
        Arrays contendo as sequencias X (velocidade) e Y (pressão)
    """

    X = None
    Y = None

    calibrator = Calibrator()
    for filename in tqdm(os.listdir("../data/" + folder_path)):
        if filename.endswith(".csv"):
            df_old = pd.read_csv("../data/" + folder_path + "/" + filename)
            df = calibrator.apply_calibration(df_old)
            df = calibrator.get_corrected_dp(df)

            pt_sequences = df_to_sequence(df.PT105_corrected, window_size)
            ft_sequences = df_to_sequence(df.FT101, window_size)

            if X is None:
                X, Y = ft_sequences, pt_sequences
            else:
                X = np.concatenate((X, ft_sequences))
                Y = np.concatenate((Y, pt_sequences))
    return X, Y


def df_to_sequence(data: pd.DataFrame, window_size: int) -> np.ndarray:
    data_array = data.values.reshape(-1, 1)
    sequences = np.lib.stride_tricks.sliding_window_view(
        data_array, window_shape=(window_size, 1)
    )

    return sequences.reshape(-1, window_size, 1)


def train_test_split_ae(
    sequence: np.ndarray, test_size: float = 0.25, shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Função responsável por dividir o conjunto de dados em
    treino e validação baseado em uma fração `test_size`.

    Parameters
    ----------
    sequence: np.ndarray
        Sequência a ser dividida..

    test_size: float
        Fração que representa o tamanho do conjunto de
        validação em relação ao tamanho da sequencia de
        entrada

    shuffle: bool
        Se os conjuntos são embaralhados ou não.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Os valores na ordem Xtrain, Xtest
    """
    n_samples = len(sequence)

    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        idx = np.arange(n_samples)

    test_set_size = int(n_samples * test_size)
    test_idx = idx[:test_set_size]
    train_idx = idx[test_set_size:]

    X_train = sequence[train_idx]
    X_test = sequence[test_idx]

    return X_train, X_test


class Calibrator:
    def __init__(self):
        files = ["PT105", "FT101"]
        self.data: dict = {}
        for filename in files:
            df = pd.read_csv(
                f"../data/CALIBRACAO/{filename}.csv", delimiter=",", decimal=","
            )
            x, y = df.iloc[:, 0], df.iloc[:, 1]
            self.a, self.b = np.polyfit(x, y, 1)
            self.data[filename] = (self.a, self.b)

    def __str__(self) -> str:
        return str(self.data)

    def apply_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PT105"] = (
            self.data["PT105"][0] * df["cDAQ1Mod1/ai2"] + self.data["PT105"][1]
        )
        df["FT101"] = (
            self.data["FT101"][0] * df["cDAQ1Mod1/ai3"] + self.data["FT101"][1]
        )
        return df

    def get_corrected_dp(self, df_calibrated: pd.DataFrame) -> pd.DataFrame:
        """Função que ajusta os dados do pt105 no leito vazio
        em relação ao ft101 e retorna os coeficientes do ajuste.

        Returns:
            O dataframe ajustado com os valores corrigidos.
        """
        pt105_means = []
        ft101_means = []

        for filename in os.listdir("../data/VAZIO"):
            if filename.endswith(".csv"):
                df = pd.read_csv("../data/VAZIO/" + filename)
                df = self.apply_calibration(df)
                pt105_means.append(df["PT105"].mean())
                ft101_means.append(df["FT101"].mean())

        pt105_means = np.array(pt105_means)
        ft101_means = np.array(ft101_means)

        a, b, c = np.polyfit(ft101_means, pt105_means, 2)

        df_calibrated["PT105_corrected"] = df_calibrated["PT105"] - (
            a * df_calibrated["FT101"] ** 2 + b * df_calibrated["FT101"] + c
        )

        return df_calibrated


class MinMaxScaler_AE:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def __str__(self) -> str:
        return f"min: {self.min_val}\nmax: {self.max_val}\n"

    def fit(self, X_train):
        """
        Ajusta o scaler com base nos valores mínimos e máximos do conjunto de treino.

        Args:
            X_train (numpy array): Dados de treino.
        """
        self.min_val = np.min(X_train, axis=0)
        self.max_val = np.max(X_train, axis=0)

    def transform(self, X):
        """
        Transforma os dados com base nos mínimos e máximos calculados no treino.

        Args:
            X (numpy array): Dados a serem normalizados.

        Returns:
            X_scaled (numpy array): Dados normalizados entre 0 e 1.
        """
        return (X - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return X_scaled * (self.max_val - self.min_val) + self.min_val
