import matplotlib.pyplot as plt
import numpy as np


def init_graph():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 26
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams['lines.markersize'] = 12


def plt_color_list() -> list:
    return ["r", "g", "b", "c", "m", "y", "k", "w"]


def new_ber_canvas(xlabel: str, x_min: int = 0, x_max: int = 25, y_min_order: int = -6, y_max_order: int = 0) -> (plt.Figure, plt.Axes):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    y_min = pow(10, y_min_order)
    y_max = pow(10, y_max_order)

    yticks = [10 ** y_order for y_order in range(y_min_order, y_max_order+1)]
    ax.set_yticks(yticks)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(linestyle='--')
    return fig, ax


def new_snr_ber_canvas(snr_min: int = 0, snr_max: int = 25, y_min_order: int = -6, y_max_order: int = 0) -> (
        plt.Figure, plt.Axes):
    # TODO　後方互換のために残してある．このメソッドは無くすべき．
    fig, ax = new_ber_canvas("SNR [dB]", snr_min, snr_max, y_min_order, y_max_order)
    ax.set_xticks(np.arange(snr_min, snr_max + 1, 5.0))
    return fig, ax


def new_learning_curve_canvas(epochs: int) -> (plt.Figure, plt.Axes):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    ax.grid(which='major', alpha=0.25)
    ax.set_xticks(range(1, epochs, 2))
    ax.set_xlim(0, epochs)
    return fig, ax
