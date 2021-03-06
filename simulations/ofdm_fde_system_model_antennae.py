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
    phi: float
    antennae_min: int
    antennae_max: int
    antennae_num: int
    gamma: float
    PA_IBO: int
    PA_rho: int
    LNA_IBO: int
    LNA_rho: int
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
    standardization: bool
    seed: bool
    delay: int
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
    SIMULATIONS_NAME = 'ofdm_fde_system_model_antennae'

    params_dict, output_dir = settings.init_simulation(SIMULATIONS_NAME, ofdm=True)
    params = Params.from_params_dict(params_dict)

    antennae_array = np.linspace(params.antennae_min, params.antennae_max, params.antennae_num, dtype=int)

    errors = np.zeros((params.antennae_num, params.trials))
    losss = np.zeros((params.antennae_num, params.trials, params.nEpochs))
    val_losss = np.zeros((params.antennae_num, params.trials, params.nEpochs))

    for trials_index in tqdm(range(params.trials)):
        h_si = []
        h_s = []

        for _ in range(params.antennae_min, params.antennae_max+1):
            h_si.append(m.exponential_decay_channel(1, params.h_si_len))
            h_s.append(m.exponential_decay_channel(1, params.h_s_len))

        for antennae_array_index, antennae in enumerate(antennae_array):
            antennae_h_si = h_si[:antennae]
            antennae_h_s = h_s[:antennae]

            sigma = m.sigmas(params.SNR)
            sigma = sigma * np.sqrt(antennae)

            nn_model = simulation(
                params.block,
                params.subcarrier,
                params.CP,
                sigma,
                params.gamma,
                params.phi,
                params.PA_IBO,
                params.PA_rho,
                params.LNA_IBO,
                params.LNA_rho,
                antennae_h_si,
                antennae_h_s,
                params.h_si_len,
                params.h_s_len,
                antennae,
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

            errors[antennae_array_index][trials_index] = nn_model.error
            losss[antennae_array_index][trials_index][:] = nn_model.history.history['loss']
            val_losss[antennae_array_index][trials_index][:] = nn_model.history.history['val_loss']

    result = Result(params, errors, losss, val_losss)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_ber_canvas("antennae", params.antennae_min, params.antennae_max)
    n_sum = params.test_bits * params.trials
    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(antennae_array, bers, color="k", marker='o', linestyle='--', label="OFDM")

    plt.tight_layout()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    # slack通知用
    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    for gamma_index, gamma in enumerate(antennae_array):
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params.nEpochs)
        loss_avg = np.mean(losss[gamma_index], axis=0).T
        val_loss_avg = np.mean(val_losss[gamma_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
        learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.savefig(output_dir + '/snr_db_' + str(gamma) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params.to_dict(), output_dir, output_png_path)
