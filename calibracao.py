import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.metrics import r2_score
import os

from utils.helper import Calibrator

plt.style.use(["science", "ieee", "notebook"])

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (9, 4)


def fit_and_plot(filename: str) -> tuple[float, float]:
    """
    Função que ajusta e plota os gráficos

    :param filename: Nome do arquivo .csv que contem
    os dados para ajuste
    """

    df = pd.read_csv(f"data/calibracao/{filename}.csv", delimiter=",", decimal=",")
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    a, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.plot(x, a * x + b, ls="--", c="grey")
    ax.scatter(x, y, marker="s", c="k")

    textstr: str = "\n".join(
        (
            rf"$(-\Delta P)$ = {a:.2f}$i$ + {b:.2f}",
            f"$R^2$ = {r2_score(y, a*x + b):.4f}",
        )
    )

    props = {"boxstyle": "square", "facecolor": "white"}
    ax.text(
        0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props
    )
    ax.set_xlabel("Corrente (mA)")
    ax.set_ylabel("Pressão (kPa)")

    plt.savefig(f"images/{filename}.pdf", dpi=300, bbox_inches="tight")

    print(f"SENSOR: {filename}\tAjuste: y = {a:.3f}*x + {b:.3f}")

    return a, b

def get_corrected_dp():
    calibrator = Calibrator()
    pt105_medias = []
    ft101_medias = []
    for filename in os.listdir("data/VAZIO"):
        if filename.endswith(".csv"):
            df = pd.read_csv("data/VAZIO/"+filename)
            df = calibrator.apply_calibration(df)
            pt105_medias.append(df["PT105"].mean())
            ft101_medias.append(df["FT101"].mean())

    pt105_medias = np.array(pt105_medias)
    ft101_medias = np.array(ft101_medias)

    a, b, c = np.polyfit(ft101_medias, pt105_medias, 2)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(ft101_medias, pt105_medias)
    ax.plot(ft101_medias, a*ft101_medias**2 + b*ft101_medias + c, c="r")
    ax.set_xlabel("FT101 (kPa)")
    ax.set_ylabel("PT105 no leito vazio (kPa)")

    plt.savefig("images/ajuste-vazio.pdf", dpi=300, bbox_inches="tight")
    print(a, b, c)
    return None


fit_and_plot("FT101")
fit_and_plot("PT105")
get_corrected_dp()