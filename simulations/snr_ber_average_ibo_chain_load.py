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

    dirname = settings.dirname_current_datetime(SIMULATIONS_NAME)
    settings.init_output(dirname)

    param_path = "../results/snr_ber_average_ibo/2020/12/03/15_45_12/params.json"
    params = settings.load_param(param_path)

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])
    n_sum = params["test_bits"] * params['SNR_AVERAGE'] * load_files
    fig, ax = graph.new_snr_ber_canvas(params['SNR_MIN'], params['SNR_MAX'])

    pkl_paths = ["../results/snr_ber_average_ibo/2020/12/03/15_45_12/result.pkl"]

    draw_snr_ber(ax, snrs_db, n_sum, pkl_paths, 'test', 'k')

    ax.legend(fontsize=12)
    plt.savefig(dirname + '/SNR_BER.pdf')