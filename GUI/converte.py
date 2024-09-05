import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data = pd.read_csv("teste.csv", delimiter=",")

# Ler os dados 
PT103 = data["cDAQ1Mod1/ai1"]
FT101 = data["cDAQ1Mod1/ai3"]
TT101 = data["cDAQ1Mod1/ai4"]

# Converte os dados para unidades de engenharia 
PT103 = PT103*1e-3*173593 - 701.12
FT101 = FT101*1e-3*3125000 - 12500
TT101 = TT101*6250*1e-3 - 25

# Filtra os dados

PT103_filtered = savgol_filter(PT103, window_length=20, polyorder=4)
FT101_filtered = savgol_filter(FT101, window_length=20, polyorder=4)
TT101_filtered = savgol_filter(TT101, window_length=20, polyorder=4)

# Visualiza os dados
plt.plot(PT103, 'k--')
plt.plot(PT103_filtered, 'b-')
plt.show()

plt.plot(FT101, 'k--')
plt.plot(FT101_filtered, 'g-')
plt.show()

# Curva de fluidização?
plt.plot(FT101, PT103, marker="o", ls='', c="k")
plt.ylabel(r"$-\Delta P$ (Pa)")
plt.title("Curva de Fluidização")
plt.show()

# Temperatura?
plt.plot(TT101)
plt.plot(TT101_filtered, 'r-')
plt.title("Temperatura (°C)")
plt.show()
