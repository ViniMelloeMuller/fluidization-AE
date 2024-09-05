import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nidaqmx.constants import AcquisitionType

plt.style.use("seaborn-v0_8-dark-palette")


SENSORES = ["cDAQ1Mod1/ai1", "cDAQ1Mod1/ai3", "cDAQ1Mod1/ai4"]
SENSORES_LABEL = ["PT103", "FT101", "TT101"]

# ai1 = PT103
# ai3 = FT101
# ai4 = TEMPERATURA

SAMPLE_RATE = 20  # Taxa de amostragem (Hz)
UPDATE_INTERVAL = 30  # Intervalo de atualização do gráfico (ms)


class GUI:
    def __init__(self, root, sensores) -> None:
        self.sensores = sensores
        self.data_buffer = []
        self.plotter = None

        self.root = root
        self.root.title("Visualização dos dados de Sensores")
        self.root.geometry("1200x600")

        self.root.resizable(0, 0)

        self.root.tk.call("source", "forest-light.tcl")
        self.style = ttk.Style()
        self.style.theme_use("forest-light")

        self.create_widgets()

    def create_widgets(self) -> None:
        sensor_frame = ttk.LabelFrame(
            self.root, text="Selecione os sensores", padding=10
        )
        sensor_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.sensor_vars = {
            sensor: tk.BooleanVar(value=True) for sensor in self.sensores
        }

        for sensor in self.sensores:
            ttk.Checkbutton(
                sensor_frame, text=sensor, variable=self.sensor_vars[sensor]
            ).pack()

        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        ttk.Button(button_frame, text="Inicializar", command=self.start_plot).pack(
            fill="x", pady=2
        )
        ttk.Button(button_frame, text="Reiniciar Dados", command=self.reset_plot).pack(
            fill="x", pady=2
        )

        self.filename_entry = ttk.Entry(button_frame, justify="center")
        self.filename_entry.pack(fill="x", pady=2)
        self.filename_entry.insert(0, "teste")  # Valor padrão

        ttk.Button(button_frame, text="Exportar Dados", command=self.export_data).pack(
            fill="x", pady=2
        )

        self.plot_frame = ttk.Frame(self.root, width=1000, height=580)
        self.plot_frame.grid(
            row=0, column=1, rowspan=5, padx=10, pady=10, sticky="nsew"
        )
        self.plot_frame.grid_propagate(False)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        placeholder_label = ttk.Label(self.plot_frame, text="Gráfico será exibido aqui")
        placeholder_label.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

    def start_plot(self) -> None:
        selected_sensors: list = [
            sensor for sensor in self.sensores if self.sensor_vars[sensor].get()
        ]
        if not selected_sensors:
            messagebox.showwarning("Aviso", "Nenhum sensor selecionado!")
            return

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.xdata: list = []
        self.ydata: dict = {sensor: [] for sensor in selected_sensors}
        self.lines: dict = {
            sensor: self.ax.plot([], [], lw=2, label=sensor)[0]
            for sensor in selected_sensors
        }

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(3, 25)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Corrente (mA)")
        self.ax.legend()

        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.get_tk_widget().config(
            width=self.plot_frame.winfo_width(), height=self.plot_frame.winfo_height()
        )

        self.plotter = canvas

        # Rotina para aquisição de dados

        self.task = nidaqmx.Task()
        for sensor in selected_sensors:
            self.task.ai_channels.add_ai_current_chan(sensor)

        self.task.timing.cfg_samp_clk_timing(
            SAMPLE_RATE, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=2
        )

        self.update_plot()

    def update_plot(self) -> None:
        if self.task:
            try:
                frame: int = len(self.xdata)
                data = self.task.read(number_of_samples_per_channel=2)
                # self.xdata.append(frame)
                for sample_index in range(len(data[0])):
                    self.xdata.append(frame + sample_index)
                    for i, sensor in enumerate(self.ydata.keys()):
                        self.ydata[sensor].append(
                            data[i][0] * 1e3
                        )  # Convertendo para mA

                for sensor in self.ydata:
                    self.lines[sensor].set_data(self.xdata, self.ydata[sensor])

                self.ax.set_xlim(
                    max(0, frame + len(data[0]) - 100), frame + len(data[0])
                )
                self.plotter.draw()

                self.root.after(UPDATE_INTERVAL, self.update_plot)

            except Exception as e:
                messagebox.showerror(
                    "ERRO", f"Ocorreu um erro durante a coleta de dados: {e}"
                )
                self.task = None

    def reset_plot(self) -> None:
        if self.plotter:
            self.xdata: list = []
            self.ydata: dict = {sensor: [] for sensor in self.ydata.keys()}
            for line in self.lines.values():
                line.set_data([], [])
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(3, 25)
            self.plotter.draw()

    def export_data(self) -> None:
        folderpath: str = "C:/Users/LPMP/Desktop/Vinicius/Dados"
        filename: str = self.filename_entry.get()
        path: str = folderpath + "/" + filename + ".csv"
        try:
            df: pd.DataFrame = pd.DataFrame(self.ydata)
            df.to_csv(path)
            messagebox.showinfo("Sucesso", f"Dados Exportados com sucesso para {path}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao Exportar os dados: {e}")

        self.reset_plot()


if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root, SENSORES)
    root.mainloop()
