import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def folder_to_sequence(folder_path: str, window_size: int) -> np.ndarray:
    """
    Função usada para converter os dados .csv de uma pasta
    para o formato aceito pelo autocodificador LSTM.

    Parameters
    ----------
    folder_path: str
        Caminho (pasta) em que os dados estão localizados.

    window_size: int
        Tamanho da janela de dados que o modelo receberá.

    Returns
    -------
    np.ndarray
        Array contendo as sequencias na forma de sequencias
    """

    X = None

    calibrator = Calibrator()
    for filename in tqdm(os.listdir("data/" + folder_path)):
        if filename.endswith(".csv"):
            df_old = pd.read_csv("data/" + folder_path + "/" + filename)
            df = calibrator.apply_calibration(df_old)
            sequences = df_to_sequence(df.PT105, window_size)
            if X is None:
                X = sequences
            else:
                X = np.concatenate((X, sequences))
    return X


def df_to_sequence(data: pd.DataFrame, window_size: int) -> np.ndarray:
    x = []
    for i in range(len(data) - window_size):
        row = [[r] for r in data[i : i + window_size]]
        x.append(row)

    return np.array(x)


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
        files = ["PT105"]
        self.data: dict = {}
        for filename in files:
            df = pd.read_csv(
                f"data/calibracao/{filename}.csv", delimiter=",", decimal=","
            )
            x, y = df.iloc[:, 0], df.iloc[:, 1]
            self.a, self.b = np.polyfit(x, y, 1)
            self.data[filename] = (self.a, self.b)

    def __str__(self) -> str:
        return str(self.data)

    def apply_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        df["PT105"] = self.a * df["cDAQ1Mod1/ai2"] + self.b
        return df


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


if __name__ == "__main__":
    TF_ENABLE_ONEDNN_OPTS = 0
    print("Hellooooo!")
