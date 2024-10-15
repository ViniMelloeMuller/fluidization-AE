import pandas as pd
import os
import matplotlib.pyplot as plt

FTMEDIO = []
PTMEDIO = []
for filename in os.listdir():
    if filename.endswith(".csv"):
        df = pd.read_csv(filename, delimiter=",")
        FTMEDIO.append(df.loc[:, "cDAQ1Mod1/ai3"].mean())
        PTMEDIO.append(df.loc[:, "cDAQ1Mod1/ai2"].mean())

dataframe = {"FTMEDIO": FTMEDIO, "PTMEDIO": PTMEDIO}
df = pd.DataFrame(dataframe)
df.to_excel("MEDIAS.xlsx")
