# from simulations.previous_research.previous_research import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph

class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray

    def __init__(self, params, errors, losss, val_losss):
        self.params = params
        self.errors = errors
        self.losss = losss
        self.val_losss = val_losss


def load_pkl_file(pkl_path: str) -> Result:
    with open(pkl_path, 'rb') as f:
        logging.info("loaded_pkl: %s" % pkl_path)
        return pickle.load(f)


def draw_snr_ber(ax: plt.Axes, snrs_db: np.ndarray, n_sum: int, pkl_paths: list, label: str = 'SNR_BER', color: str = 'k'):
    results = map(load_pkl_file, pkl_paths)

    # 結合させる
    errors_list = []

    for result in results:
        errors_list.append(result.errors)

    errors = np.concatenate(errors_list, 1)

    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color=color, marker='o', linestyle='--', label=label)


if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research_non_selective_lin'
    load_files = 1 # 同じ条件で読み込む数

    # dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    dirname = "../results/keep/previous_research"
    settings.init_output(dirname)

    param_path = "../results/keep/previous_research_non_selective/02_10_31/params.json"
    params = settings.load_param(param_path)
    # n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files
    n_sum = 3998 * params['SNR_AVERAGE']

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    fig, ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])

    pkl_paths = ["../results/keep/previous_research_non_selective/02_10_31/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'non lin', 'r')

    pkl_paths = ["../results/keep/previous_research_non_selective_lin/18_01_07/result.pkl"]
    n_sum = 4000 * params['SNR_AVERAGE']
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'lin', 'b')

    pkl_paths = ["../results/keep/previous_research_non_selective_non_cancell/18_08_32/result.pkl"]
    n_sum = 4000 * params['SNR_AVERAGE']
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'without cancell', 'k')

    ax.legend(fontsize=12)
    plt.savefig(dirname + '/SNR_BER.pdf')