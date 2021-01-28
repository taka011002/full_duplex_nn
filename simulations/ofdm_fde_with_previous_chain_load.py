# from simulations.previous_research.previous_research import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph
import os
from ofdm_fde_with_previous import Result

def load_pkl_file(pkl_path: str) -> Result:
    with open(pkl_path, 'rb') as f:
        logging.info("loaded_pkl: %s" % pkl_path)
        return pickle.load(f)


def draw_snr_ber(ax: plt.Axes, snrs_db: np.ndarray, n_sum: int, pkl_path: str, label: str = 'SNR_BER',
                 color: str = 'k'):
    result = load_pkl_file(pkl_path)

    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color=color, marker='o', linestyle='--', label=label)


if __name__ == '__main__':
    SIMULATIONS_NAME = os.path.basename(__file__).split('.')[0]
    load_files = 1  # 同じ条件で読み込む数

    # dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    dirname = "../results/keep/ofdm_fde_system_model_chain_load"
    settings.init_output(dirname)

    param_path = "../results/ofdm_fde_system_model/2021/01/22/01_25_54/params.json"
    params = settings.load_param(param_path)
    # n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files
    n_sum = 7680 * params['SNR_AVERAGE']

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    fig, ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'], -5)
    ax.set_yticks([10**0, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5])

    pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_24_35/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='^', linestyle='-', label=r"$I=1, k=0$", ms=12)

    pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_24_29/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='^', linestyle='--', label=r"$I=1, k=4$", ms=12, markerfacecolor='None')

    pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_24_13/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='o', linestyle='-', label=r"$I=2, k = 0$", ms=12)

    pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_23_49/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='o', linestyle='--', label=r"$I=2, k = 4$", ms=12, markerfacecolor='None')

    pkl_path = "../results/ofdm_fde_system_model/2021/01/22/01_25_54/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='d', linestyle='-', label=r"$I=3, k = 0$", ms=12)

    pkl_path = "../results/ofdm_fde_system_model/2021/01/22/01_25_45/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='d', linestyle='--', label=r"$I=3, k = 4$", ms=12, markerfacecolor='None')

    ax.legend(fontsize=19, loc=3)
    plt.savefig(dirname + '/SNR_BER.eps', bbox_inches='tight')
    plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')

    plt.show()