import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import savgol_filter


def read_data(folder_path: str) -> list:
    """
    Função que lê os dados de uma determinada pasta e os adiciona a
    uma lista

    Parameters
    ----------
    folder_path
        o caminho para a pasta contendo os dados

    Returns
    -------
    list
        lista contendo os dados na forma de DataFrames
    """
    dataframes: list = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            dataframes.append(df)

    return dataframes


def convert_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faz o processamento de um DataFrame cru

    Pega um dataframe cru (com os dados na forma de correente) e transforma em unidades
    de engenharia pelas curvas de calibração + filtra usando o filtro de savitz golay
    e por fim determina a velocidade pela teoria da obstrução de Bernoulli.

    Parameters
    ----------
    df
        O dataframe de entrada (corrente)

    Returns
    -------
    pd.DataFrame
        o dataframe de saída (unidades legais)
    """

    # Le os dados
    PT103 = df["cDAQ1Mod1/ai1"]
    FT101 = df["cDAQ1Mod1/ai3"]
    TT101 = df["cDAQ1Mod1/ai4"]

    # Converte os dados para unidades de engenharia
    PT103 = PT103 * 1e-3 * 173593 - 701.12  # Pa
    FT101 = FT101 * 1e-3 * 3125000 - 12500  # Pa
    TT101 = TT101 * 6250 * 1e-3 - 25

    # Filtra os dados
    PT103_filtered = savgol_filter(PT103, window_length=50, polyorder=4)
    FT101_filtered = savgol_filter(FT101, window_length=50, polyorder=4)
    TT101_filtered = savgol_filter(TT101, window_length=50, polyorder=4)

    # Determina velocidade do Gas através da queda de pressão
    ## Chute inicial, super imprecisa, arrumar depois!!!!! 410
    tmed = np.mean(TT101_filtered)
    rho_f = 101325 / 287 / (tmed + 273.15)  # kg/m3
    D = 0.1  # m
    D2 = dt = 12.3e-3  # m

    # Chute inicial -> Não considera efeitos de atrito
    u20 = (2 * FT101_filtered / (rho_f * (1 - D2**4 / D**4))) ** (1 / 2)  # m/s?

    # Cria um novo DataFrame com os dados convertidos
    df_new = {
        "PT103": PT103_filtered,  # Pressão em Pa
        "FT101": FT101_filtered,  # Pressão também em Pa
        "TT101": TT101_filtered,  # Temperatura em °C
        "Velocidade": u20,  # Velocidade do gás em m/s
    }

    return pd.DataFrame(df_new)


if __name__ == "__main__":
    datalist: list = read_data("data/VAZIO")
    mean_DP = []
    mean_u = []

    for df in tqdm(datalist):
        df_new = convert_dataframe(df)
        dP_med = df_new.loc[:, "PT105"].mean()
        u_med = df_new.loc[:, "Velocidade"].mean()
        mean_DP.append(dP_med)
        mean_u.append(u_med)
        print(u_med, "\t", dP_med)

    # Resultados muito estranhos
    plt.scatter(mean_u, mean_DP)
    plt.show()
