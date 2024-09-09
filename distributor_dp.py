import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.signal import savgol_filter

def read_data(folder_path: str) -> list:
    """Function that reads all the .csv files from a folder

    Returns:
        list: A list of dataframes, one for each excel file found 
        in the folder
    """
    dataframes: list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            dataframes.append(df)

    return dataframes

def convert_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Le os dados
    PT103 = df["cDAQ1Mod1/ai1"]
    FT101 = df["cDAQ1Mod1/ai3"]
    TT101 = df["cDAQ1Mod1/ai4"]

    # Converte os dados para unidades de engenharia
    PT103 = PT103 * 1e-3 * 173593 - 701.12 #Pa
    FT101 = FT101 * 1e-3 * 3125000 - 12500 #Pa
    TT101 = TT101 * 6250 * 1e-3 - 25

    # Filtra os dados
    PT103_filtered = savgol_filter(PT103, window_length=50, polyorder=4)
    FT101_filtered = savgol_filter(FT101, window_length=50, polyorder=4)
    TT101_filtered = savgol_filter(TT101, window_length=50, polyorder=4)

    # Determina velocidade do Gas através da queda de pressão
    dPo = FT101_filtered / 98.066 #Pa -> CmH2O
    patm = 762.81 #mmHg
    tmed = np.mean(TT101_filtered) #oC
    dor = 12.3/10 #cm
    dt = 10  #cm
    Aor = np.pi*(dor)**2/4
    At = np.pi*(dt)**2/4
    m = Aor / At
    beta = 0.3041 + 0.0876*m - 0.1166*m**2 + 0.4089*m**3
    lambda_f = 1 - (beta*dPo)/(patm)
    alpha = 0.5959 + 0.0312*m**1.05 - 0.184*m**4
    Q = 0.0573*lambda_f*alpha*Aor*np.sqrt(dPo/(1 - m**2) * patm/(273 + TT101_filtered)) #kg/min
    rho = (patm*133.3 - FT101_filtered)/(287*(tmed + 273)) #kg/m3
    F = Q/rho #m3/min 
    u = (F/(At*1e-2)/60)*100 #cm/s

    # Cria um novo DataFrame com os dados convertidos
    df_new = {
        "PT103": PT103_filtered,
        "FT101": FT101_filtered,
        "TT101": TT101_filtered,
        "Velocidade": u,
    }

    return pd.DataFrame(df_new)

if __name__ == "__main__":

    datalist: list = read_data("data/VAZIO")
    mean_DP = []
    mean_u = []

    for df in tqdm(datalist):
        df_new = convert_dataframe(df)
        dP_med = df_new.loc[:, "PT103"].mean()
        u_med = df_new.loc[:, "Velocidade"].mean()
        mean_DP.append(dP_med)
        mean_u.append(u_med)
        print(u_med, "\t", dP_med)

    # Resultados muito estranhos
    plt.scatter(mean_u, mean_DP)
    plt.show()

