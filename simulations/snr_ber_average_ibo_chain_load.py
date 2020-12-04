from simulations.snr_ber_average_ibo import Result  # pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
from simulations.common import settings
from simulations.common import graph


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

    errors = np.concatenate(errors_list, 2)

    IBO_index = 0  # 現状IBOを変更させずにシミュレーションしているので決めうち
    errors_sum = np.sum(errors[IBO_index], axis=1)
    bers = errors_sum / n_sum
    np.place(bers, bers == 0, None)
    ax.plot(snrs_db, bers, color=color, marker='o', linestyle='--', label=label)


if __name__ == '__main__':
    SIMULATIONS_NAME = 'snr_ber_average_ibo_chain_load'
    load_files = 1 # 同じ条件で読み込む数

    # dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    dirname = "../results/keep/frequency_selective/ch_5_delay_graph"
    settings.init_output(dirname)

    param_path = "../results/keep/frequency_selective/ch_5_anthena_1/params.json"
    params = settings.load_param(param_path)
    n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    fig, ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_1/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 1', 'r')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_1_delay/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 1(delay)', 'g')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_2/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 2', 'b')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_2_delay/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 2(delay)', 'c')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_3/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 3', 'm')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_3_delay/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 3(delay)', 'y')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_4/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 4', 'k')

    pkl_paths = ["../results/keep/frequency_selective/ch_5_anthena_4_delay/result.pkl"]
    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'receive_antenna: 4(delay)', '#a65628')

    ax.legend(fontsize=12)
    plt.savefig(dirname + '/SNR_BER.pdf')