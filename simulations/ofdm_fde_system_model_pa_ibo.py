from __future__ import annotations
from src import modules as m
import numpy as np
from simulations.common import graph
from simulations.common import settings
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataclasses
import pickle
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from src.ofdm_simulation import simulation


@dataclass_json
@dataclass
class Params:
    trials: int
    block: int
    subcarrier: int
    CP: int
    h_si_len: int
    h_s_len: int
    SNR: int
    equalizer: int
    gamma: float
    phi: float
    PA_IBO_MIN: int
    PA_IBO_MAX: int
    PA_IBO_NUM: int
    PA_rho: int
    LNA_IBO: int
    LNA_rho: int
    receive_antenna: int
    TX_IQI: bool
    PA: bool
    LNA: bool
    RX_IQI: bool
    nHidden: List[int]
    nEpochs: int
    optimizer: str
    learningRate: float
    momentum: float
    trainingRatio: float
    batchSize: int
    delay: int
    standardization: bool
    seed: bool
    test_bits: int

    @classmethod
    def from_params_dict(cls, params: dict) -> Params:
        """
        dataclass_jsonで用意されているfrom_dictをそのまま呼ぶと型補完が効かない．
        明示的に書くことにより型補完が効くようにする．

        :param params:
        :return:
        """
        return cls.from_dict(params)


@dataclasses.dataclass
class Result:
    params: Params
    errors: np.ndarray
    losss: np.ndarray
    val_losss: np.ndarray


if __name__ == '__main__':
    SIMULATIONS_NAME = 'ofdm_fde_system_model_pa_ibo'

    params_dict, output_dir = settings.init_simulation(SIMULATIONS_NAME, ofdm=True)
    params = Params.from_params_dict(params_dict)

    PA_IBO_array = np.linspace(params.PA_IBO_MIN, params.PA_IBO_MAX, params.PA_IBO_NUM)

    sigma = m.sigmas(params.SNR)
    sigma = sigma * np.sqrt(params.receive_antenna)
    errors = np.zeros((params.PA_IBO_NUM, params.trials))
    losss = np.zeros((params.PA_IBO_NUM, params.trials, params.nEpochs))
    val_losss = np.zeros((params.PA_IBO_NUM, params.trials, params.nEpochs))

    for trials_index in tqdm(range(params.trials)):
        h_si = []
        h_s = []

        for i in range(params.receive_antenna):
            h_si.append(m.exponential_decay_channel(1, params.h_si_len))
            h_s.append(m.exponential_decay_channel(1, params.h_s_len))

        for PA_IBO_array_index, PA_IBO in enumerate(PA_IBO_array):
            nn_model = simulation(
                params.block,
                params.subcarrier,
                params.CP,
                sigma,
                params.gamma,
                params.phi,
                PA_IBO,
                params.PA_rho,
                params.LNA_IBO,
                params.LNA_rho,
                h_si,
                h_s,
                params.h_si_len,
                params.h_s_len,
                params.receive_antenna,
                params.TX_IQI,
                params.PA,
                params.LNA,
                params.RX_IQI,
                params.nHidden,
                params.optimizer,
                params.learningRate,
                params.momentum,
                params.trainingRatio,
                params.nEpochs,
                params.batchSize,
                params.delay,
                params.standardization
            )

            errors[PA_IBO_array_index][trials_index] = nn_model.error
            losss[PA_IBO_array_index][trials_index][:] = nn_model.history.history['loss']
            val_losss[PA_IBO_array_index][trials_index][:] = nn_model.history.history['val_loss']

    result = Result(params, errors, losss, val_losss)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_ber_canvas("PA_IBO [dB]", params.PA_IBO_MIN, params.PA_IBO_MAX,)
    n_sum = params.test_bits * params.trials
    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(PA_IBO_array, bers, color="k", marker='o', linestyle='--', label="OFDM")

    plt.tight_layout()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    # slack通知用
    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    for PA_IBO_index, PA_IBO in enumerate(PA_IBO_array):
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params.nEpochs)
        loss_avg = np.mean(losss[PA_IBO_index], axis=0).T
        val_loss_avg = np.mean(val_losss[PA_IBO_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
        learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.savefig(output_dir + '/snr_db_' + str(PA_IBO) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params.to_dict(), output_dir, output_png_path)
