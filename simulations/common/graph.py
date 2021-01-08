import matplotlib.pyplot as plt
import numpy as np


def init_graph():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 26
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plt_color_list() -> list:
    return ["r", "g", "b", "c", "m", "y", "k", "w"]


def new_snr_ber_canvas(snr_min: int = 0, snr_max: int = 25, y_min_order: int = -6, y_max_order: int = 0) -> (
        plt.Figure, plt.Axes):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    ax.set_xlim(snr_min, snr_max)
    y_min = pow(10, y_min_order)
    y_max = pow(10, y_max_order)

    ax.set_xticks(np.arange(snr_min, snr_max+1, 5.0))
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(snr_min, snr_max)
    ax.grid(linestyle='--')
    return fig, ax


def new_learning_curve_canvas(epochs: int) -> (plt.Figure, plt.Axes):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('less')
    ax.set_yscale('log')
    ax.grid(which='major', alpha=0.25)
    ax.set_xticks(range(1, epochs, 2))
    ax.set_xlim(0, epochs)
    return fig, ax
