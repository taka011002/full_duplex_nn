# from simulations.previous_research.previous_research import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph
import dataclasses

@dataclasses.dataclass
class Result:
    params: dict
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray
    error_array: np.ndarray
    previous_losss: np.ndarray
    previous_val_losss: np.ndarray
    lin_error_array: np.ndarray
    non_cancell_error_array: np.ndarray


def load_pkl_file(pkl_path: str) -> Result:
    with open(pkl_path, 'rb') as f:
        logging.info("loaded_pkl: %s" % pkl_path)
        return pickle.load(f)


def draw_snr_ber(ax: plt.Axes, snrs_db: np.ndarray, n_sum: int, pkl_path: str, label: str = 'SNR_BER', color: str = 'k'):
    result = load_pkl_file(pkl_path)

    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color=color, marker='o', linestyle='--', label=label)


if __name__ == '__main__':
    SIMULATIONS_NAME = 'snr_ber_average_ibo_with_previous_chain_load'
    load_files = 1 # 同じ条件で読み込む数

    # dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    dirname = "../results/keep/snr_ber_average_ibo_with_previous_chain_load"
    settings.init_output(dirname)

    param_path = "../results/snr_ber_average_ibo_with_previous/18_07_00/params.json"
    params = settings.load_param(param_path)
    # n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files
    n_sum = 3998 * params['SNR_AVERAGE']

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    fig, ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'], -5)

    pkl_path = "../results/snr_ber_average_ibo_with_previous/18_07_00/result.pkl"
    # draw_snr_ber(ax, snrs_db, n_sum, pkl_path, 'non lin[7db]', 'k')
    result = load_pkl_file(pkl_path)

    errors_sum = np.sum(result.non_cancell_error_array, axis=1)
    bers = errors_sum / (4000 * params['SNR_AVERAGE'])
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='*', linestyle=':', label='w/o canceller')

    errors_sum = np.sum(result.error_array, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='x', linestyle='--', label='Conventional [1]')

    errors_sum = np.sum(result.errors, axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color='k', marker='o', linestyle='-', label='Proposed')

    ax.legend(fontsize=24)
    plt.savefig(dirname + '/SNR_BER.pdf')
    plt.show()