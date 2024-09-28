import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from sklearn.metrics import r2_score

plt.style.use(["science", "ieee", "notebook"])

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (9, 4)


def fit_and_plot(filename: str) -> None:
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


fit_and_plot("FT101")
fit_and_plot("PT105")
