from src import modules as m
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import src.previous_research.fulldeplex as fd
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np
from tqdm import tqdm


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
        'SNR_AVERAGE': 10,

        "h_si_len": 2,
        "h_s_len": 2,

        "n_hidden": 10,
        "learning_rate": 0.001,
        "training_ratio": 0.9,
        "batch_size": 32,
        "nEpochs": 20,
        "L_w": 10
    }

    # データを生成する
    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    loss_array = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    val_loss_array = np.zeros((params['SNR_NUM'], params['SNR_AVERAGE'], params['nEpochs']))
    error_array = np.zeros((len(snrs_db), params['SNR_AVERAGE']))

    for trials_index in tqdm(range(params['SNR_AVERAGE'])):
        h_si = m.channel(1, params['h_si_len'])
        h_s = m.channel(1, params['h_s_len'])

        for sigma_index, sigma in enumerate(sigmas):
            system_model = PreviousSystemModel(
                sigma,
                params['gamma'],
                params['phi'],
                params['PA_IBO_dB'],
                params['PA_rho'],
                params['LNA_IBO_dB'],
                params['LNA_rho'],
                h_si,
                params['h_si_len'],
                h_s,
                params['h_s_len'],
            )
            system_model.transceive_si(
                params['n']
            )

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

            loss_array[sigma_index][trials_index][:] = previous_nn_model.history.history['loss']
            val_loss_array[sigma_index][trials_index][:] = previous_nn_model.history.history['val_loss']

            training_samples = int(np.floor(params['n'] * params['training_ratio']))
            test_n = params['n'] - training_samples

            system_model.transceive_s(
                test_n,
            )

            h_s_len = params['h_s_len']
            L_h = params['h_s_len'] - 1
            L_w = params['L_w']

            h_s_hat = fd.ls_estimation(system_model.x[0:int(test_n / 2)], system_model.y, h_s_len)
            h_s_hat = h_s_hat.reshape(1, 2)
            H = fd.toeplitz_chanel(h_s_hat.T, h_s_len, L_w)
            W = fd.mmse(H, sigma**2)

            system_model.transceive_si_s(
                test_n,
            )

            previous_nn_model.cancel(
                system_model.x[0:int(test_n / 2)],
                system_model.y,
                params['h_si_len'],
            )

            cancelled_y = previous_nn_model.cancelled_y # これが希望信号成分
            size = cancelled_y.shape[0]
            y_vec = np.array([cancelled_y[i:i + size - L_w] for i in range(L_w + 1)])

            z = np.matmul(W.conj().T, y_vec)

            d_hat = m.demodulate_qpsk(z)
            d_hat_len = d_hat.shape[0]
            error = np.sum(system_model.d_s[0:d_hat_len] != d_hat)

            error_array[sigma_index][trials_index] = error

    for sigma_index, snr_db in enumerate(snrs_db):
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])
        loss_avg = np.mean(loss_array[sigma_index], axis=0).T
        val_loss_avg = np.mean(val_loss_array[sigma_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch,
                      loss_avg,
                      color="k", marker='o',
                      linestyle='--',
                      label='Training Frame')
        learn_ax.plot(epoch,
                      val_loss_avg,
                      color="r",
                      marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.show()

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])
    n_sum = d_hat_len * params['SNR_AVERAGE']

    errors_sum = np.sum(error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(snrs_db, bers, color="k", marker='o', linestyle='--', label="MMSE")

    ber_ax.legend()
    plt.show()
