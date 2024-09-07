import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data = pd.read_csv("teste.csv", delimiter=",")

# Ler os dados
PT103 = data["cDAQ1Mod1/ai1"]
FT101 = data["cDAQ1Mod1/ai3"]
TT101 = data["cDAQ1Mod1/ai4"]

# Converte os dados para unidades de engenharia
PT103 = PT103 * 1e-3 * 173593 - 701.12 #Pa
FT101 = FT101 * 1e-3 * 3125000 - 12500 #Pa
TT101 = TT101 * 6250 * 1e-3 - 25


# Filtra os dados

PT103_filtered = savgol_filter(PT103, window_length=50, polyorder=4)
FT101_filtered = savgol_filter(FT101, window_length=50, polyorder=4)
TT101_filtered = savgol_filter(TT101, window_length=50, polyorder=4)

# Visualiza os dados
plt.plot(PT103, "k--")
plt.plot(PT103_filtered, "b-")
plt.show()

plt.plot(FT101, "k--")
plt.plot(FT101_filtered, "g-")
plt.show()

# Temperatura?
plt.plot(TT101)
plt.plot(TT101_filtered, "r-")
plt.title("Temperatura (°C)")
plt.show()

print(type(TT101_filtered))

# Determinando a velocidade do gás
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



# Curva de fluidização?
plt.plot(u, PT103_filtered, marker="o", ls="", c="k")
plt.ylabel(r"$-\Delta P$ (Pa)")
plt.title("Curva de Fluidização")
plt.show()
