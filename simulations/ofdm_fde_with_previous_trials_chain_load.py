# from simulations.previous_research.previous_research import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph
import os
from ofdm_fde_with_previous import Result
from ofdm_fde_with_previous import Params

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
    # dirname = "../results/keep/" + SIMULATIONS_NAME
    settings.init_output(dirname)

    # param_path = "../results/ofdm_fde_with_previous/2021/01/29/13_02_26/params.json"
    param_path = "../results/keep/ofdm_fde_with_previous/12_35_55/params.json"

    params_dict = settings.load_param(param_path)
    params = Params.from_params_dict(params_dict)
    n_sum = params.test_bits * params.trials

    trials_num = params.trials
    graph_x_array = np.linspace(1, params.trials, trials_num, dtype=int)
    fig, ax = graph.new_ber_canvas('trials', 0, params.trials, -5)

    # pkl_path = "../results/ofdm_fde_with_previous/2021/01/29/13_02_26/result.pkl"
    # pkl_path = "../results/ofdm_fde_system_model/2021/01/22/01_25_45/result.pkl"
    pkl_path = "../results/keep/ofdm_fde_with_previous/12_35_55/result.pkl"

    result = load_pkl_file(pkl_path)
    SNR = 25 # 使わないけど後からわかりやすいように定義しておく
    SNR_errors = result.errors[-1]

    errors_sum = np.sum(result.errors, axis=1)
    bers = np.zeros((trials_num))
    for i, trials in enumerate(graph_x_array):
        error = np.sum(SNR_errors[:trials])
        n_sum = params.test_bits * trials
        ber = error / n_sum
        bers[i] = ber

    np.place(bers, bers == 0, None)
    ax.plot(graph_x_array, bers, color="k", marker='d', linestyle='--', label="Proposal")

    # ax.legend(fontsize=19, loc=3)
    plt.savefig(dirname + '/SNR_BER.eps', bbox_inches='tight')
    plt.savefig(dirname + '/SNR_BER.pdf', bbox_inches='tight')

    plt.show()