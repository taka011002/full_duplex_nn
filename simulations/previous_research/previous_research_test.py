from src import modules as m
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np


if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research'

    params = {
        'n': 20000,  # サンプルのn数
        'gamma': 0.3,

        'phi': 3.0,
        'PA_IBO_dB': 7,
        'PA_rho': 2,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,

        'SNR_MIN': 0,
        'SNR_MAX': 25,
        'SNR_NUM': 6,
        'SNR_AVERAGE': 1,

        "h_si_len": 13,
        "n_hidden": 17,
        "learning_rate": 0.004,
        "training_ratio": 0.8,
        "batch_size": 32,
        "nEpochs": 20
    }

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    h_si = m.channel(1, params['h_si_len'])

    for sigma_index, sigma in enumerate(sigmas):
        system_model = PreviousSystemModel()
        system_model.transceive_si(
            params['n'],
            0,
            params['gamma'],
            params['phi'],
            params['PA_IBO_dB'],
            params['PA_rho'],
            params['LNA_IBO_dB'],
            params['LNA_rho'],
            h_si,
            params['h_si_len'],
        )

        plt.figure()
        plt.scatter(system_model.y.real, system_model.y.imag, color="g", label="y")
        plt.scatter(system_model.x.real, system_model.x.imag, color="red", label="x")
        plt.legend()
        plt.show()

        previous_nn_model = PreviousNNModel(
            params['h_si_len'],
            params['n_hidden'],
            params['learning_rate']
        )

        previous_nn_model.learn(
            system_model.x[0:int(params['n'] / 2)],
            system_model.y,
            params['training_ratio'],
            params['h_si_len'],
            params['nEpochs'],
            params['batch_size']
        )

        learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])
        epoch = np.arange(1, len(previous_nn_model.history.history['loss']) + 1)
        learn_ax.plot(epoch,
                      previous_nn_model.history.history['loss'],
                      color="k", marker='o',
                      linestyle='--',
                      label='Training Frame')
        learn_ax.plot(epoch,
                      previous_nn_model.history.history['val_loss'],
                      color="r",
                      marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.show()