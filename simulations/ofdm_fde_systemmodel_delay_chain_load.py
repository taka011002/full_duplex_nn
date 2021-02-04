# from simulations.previous_research.previous_research import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph
import os
from ofdm_fde_system_model_delay import Result
from ofdm_fde_system_model_delay import Params

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

    dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    # dirname = "../results/keep/ofdm_fde_system_model_chain_load"
    settings.init_output(dirname)

    param_path = "../results/keep/ofdm_fde_with_previous/12_35_55/params.json"
    params = settings.load_param(param_path)
    # n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files
    n_sum = params['test_bits'] * params['trials']

    snrs_db = np.linspace(params['graph_x_min'], params['graph_x_max'], params['graph_x_num'])
    fig, ax = graph.new_snr_ber_canvas(params['graph_x_min'], params['graph_x_max'], -4)
    ax.set_yticks([10**0, 10**-1, 10**-2, 10**-3, 10**-4])

    previous_n_sum = params['previous_test_bits'] * params['trials']
    pkl_path = "../results/keep/ofdm_fde_with_previous/12_35_55/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.non_cancell_error_array, axis=1)
    bers = errors_sum / previous_n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='x', linestyle='--', label="w/o canceller", ms=12)

    errors_sum = np.sum(result.previous_errors, axis=1)
    bers = errors_sum / previous_n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='*', linestyle='--', label="Conventional", ms=12)

    pkl_path = "../results/keep/ofdm_fde_with_previous/12_36_11/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='^', linestyle='--', label="Proposal($I = 1$)", ms=12)

    # pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_24_29/result.pkl"
    # result = load_pkl_file(pkl_path)
    # errors_sum = np.sum(result.errors, axis=1)
    # bers = errors_sum / n_sum
    # np.place(bers, bers == 0, None)
    # ax.plot(snrs_db, bers, color='k', marker='^', linestyle='--', label=r"$I=1, k=4$", ms=12, markerfacecolor='None')

    pkl_path = "../results/keep/ofdm_fde_with_previous/12_36_02/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='o', linestyle='--', label="Proposal($I = 2$)", ms=12)

    # pkl_path = "../results/ofdm_fde_system_model/2021/01/21/22_23_49/result.pkl"
    # result = load_pkl_file(pkl_path)
    # errors_sum = np.sum(result.errors, axis=1)
    # bers = errors_sum / n_sum
    # np.place(bers, bers == 0, None)
    # ax.plot(snrs_db, bers, color='k', marker='o', linestyle='--', label=r"$I=2, k = 4$", ms=12, markerfacecolor='None')

    pkl_path = "../results/keep/ofdm_fde_with_previous/12_35_55/result.pkl"
    result = load_pkl_file(pkl_path)
    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='d', linestyle='--', label=r"Proposal($I = 3$)", ms=12)

    # pkl_path = "../results/keep/ofdm_fde_with_previous/12_35_55/result.pkl"
    # result = load_pkl_file(pkl_path)
    # errors_sum = np.sum(result.errors, axis=1)
    # bers = errors_sum / n_sum
    # np.place(bers, bers == 0, None)
    # ax.plot(snrs_db, bers, color='k', marker='d', linestyle='--', label=r"$I=3, k = 4$", ms=12, markerfacecolor='None')

    ax.legend(fontsize=19, loc=3)
    plt.savefig(dirname + '/SNR_BER.eps', bbox_inches='tight')
    plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')

    plt.show()