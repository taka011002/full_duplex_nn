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
from src.previous_research.ofdm_simulation import simulation as previous_simulation
from src.ofdm_noncancel import simulation as noncancel_simulation
import os

@dataclass_json
@dataclass
class Params:
    trials: int
    block: int
    subcarrier: int
    CP: int
    h_si_len: int
    h_s_len: int
    equalizer: int
    gamma: float
    phi: float
    graph_x_min: int
    graph_x_max: int
    graph_x_num: int
    PA_IBO: int
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
    p_nHidden: List[int]
    p_optimizer: str
    p_learningRate: float
    p_batchSize: int
    p_nEpochs: int
    delay: int
    standardization: bool
    seed: bool
    train_bits: int
    test_bits: int
    previous_test_bits: int
    compensate_iqi: bool
    p_receive_antenna: int

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
    non_cancell_error_array: np.ndarray
    previous_errors: np.ndarray
    previous_losss: np.ndarray
    previous_val_losss: np.ndarray


if __name__ == '__main__':
    SIMULATIONS_NAME = os.path.basename(__file__).split('.')[0]

    params_dict, output_dir = settings.init_simulation(SIMULATIONS_NAME, ofdm=True)
    params = Params.from_params_dict(params_dict)

    graph_x_array = np.linspace(params.graph_x_min, params.graph_x_max, params.graph_x_num)

    errors = np.zeros((params.graph_x_num, params.trials))
    losss = np.zeros((params.graph_x_num, params.trials, params.nEpochs))
    val_losss = np.zeros((params.graph_x_num, params.trials, params.nEpochs))

    non_cancell_error_array = np.zeros((len(graph_x_array), params.trials))

    previous_errors = np.zeros((params.graph_x_num, params.trials))
    previous_losss = np.zeros((params.graph_x_num, params.trials, params.p_nEpochs))
    previous_val_losss = np.zeros((params.graph_x_num, params.trials, params.p_nEpochs))

    for trials_index in tqdm(range(params.trials)):
        h_si = []
        h_s = []

        for i in range(params.receive_antenna):
            h_si.append(m.exponential_decay_channel(1, params.h_si_len))
            h_s.append(m.exponential_decay_channel(1, params.h_s_len))

        for graph_x_index, SNR in enumerate(graph_x_array):
            sigma = m.sigmas(SNR)
            # previous_sigma = sigma
            previous_sigma = sigma * np.sqrt(params.p_receive_antenna)
            proposal_sigma = sigma * np.sqrt(params.receive_antenna)

            non_cancell_error_array[graph_x_index][trials_index] = noncancel_simulation(
                params.block,
                params.subcarrier,
                params.CP,
                previous_sigma,
                params.gamma,
                params.phi,
                params.PA_IBO,
                params.PA_rho,
                params.LNA_IBO,
                params.LNA_rho,
                h_si,
                h_s,
                params.h_si_len,
                params.h_s_len,
                params.TX_IQI,
                params.PA,
                params.LNA,
                params.RX_IQI,
                params.trainingRatio,
                params.compensate_iqi,
                params.p_receive_antenna
            )

            previous_nn_model = previous_simulation(
                params.block,
                params.subcarrier,
                params.CP,
                previous_sigma,
                params.gamma,
                params.phi,
                params.PA_IBO,
                params.PA_rho,
                params.LNA_IBO,
                params.LNA_rho,
                h_si,
                h_s,
                params.h_si_len,
                params.h_s_len,
                params.TX_IQI,
                params.PA,
                params.LNA,
                params.RX_IQI,
                params.p_nHidden,
                params.p_optimizer,
                params.p_learningRate,
                params.momentum,
                params.trainingRatio,
                params.p_nEpochs,
                params.p_batchSize,
                params.compensate_iqi,
                params.p_receive_antenna
            )

            previous_errors[graph_x_index][trials_index] = previous_nn_model.error
            previous_losss[graph_x_index][trials_index][:] = previous_nn_model.history.history['loss']
            previous_val_losss[graph_x_index][trials_index][:] = previous_nn_model.history.history['val_loss']

            # nn_model = simulation(
            #     params.block,
            #     params.subcarrier,
            #     params.CP,
            #     proposal_sigma,
            #     params.gamma,
            #     params.phi,
            #     params.PA_IBO,
            #     params.PA_rho,
            #     params.LNA_IBO,
            #     params.LNA_rho,
            #     h_si,
            #     h_s,
            #     params.h_si_len,
            #     params.h_s_len,
            #     params.receive_antenna,
            #     params.TX_IQI,
            #     params.PA,
            #     params.LNA,
            #     params.RX_IQI,
            #     params.nHidden,
            #     params.optimizer,
            #     params.learningRate,
            #     params.momentum,
            #     params.trainingRatio,
            #     params.nEpochs,
            #     params.batchSize,
            #     params.delay,
            #     params.standardization
            # )
            #
            # errors[graph_x_index][trials_index] = nn_model.error
            # losss[graph_x_index][trials_index][:] = nn_model.history.history['loss']
            # val_losss[graph_x_index][trials_index][:] = nn_model.history.history['val_loss']

    result = Result(params, errors, losss, val_losss, non_cancell_error_array, previous_errors, previous_losss, previous_val_losss)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_snr_ber_canvas(params.graph_x_min, params.graph_x_max, -6)

    n_sum = params.previous_test_bits * params.trials
    errors_sum = np.sum(non_cancell_error_array, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(graph_x_array, bers, color="k", marker='x', linestyle='--', label="w/o canceller")

    n_sum = params.previous_test_bits * params.trials
    errors_sum = np.sum(previous_errors, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(graph_x_array, bers, color="k", marker='d', linestyle='--', label="Previous")

    # n_sum = params.test_bits * params.trials
    # errors_sum = np.sum(errors, axis=1)
    # bers = errors_sum / n_sum
    # ber_ax.plot(graph_x_array, bers, color="k", marker='o', linestyle='--', label="Proposed")

    plt.tight_layout()
    ber_ax.legend()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    # slack通知用
    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    for graph_x_index, graph_x in enumerate(graph_x_array):
        # 従来法
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params.nEpochs)
        loss_avg = np.mean(losss[graph_x_index], axis=0).T
        val_loss_avg = np.mean(val_losss[graph_x_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
        learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.savefig(output_dir + '/' + str(graph_x) + '_NNconv.pdf', bbox_inches='tight')

        # 提案法
        learn_fig, learn_ax = graph.new_learning_curve_canvas(params.p_nEpochs)
        loss_avg = np.mean(previous_losss[graph_x_index], axis=0).T
        val_loss_avg = np.mean(previous_val_losss[graph_x_index], axis=0).T
        epoch = np.arange(1, len(loss_avg) + 1)

        learn_ax.plot(epoch, loss_avg, color="k", marker='o', linestyle='--', label='Training Frame')
        learn_ax.plot(epoch, val_loss_avg, color="r", marker='o', linestyle='--', label='Test Frame')
        learn_ax.legend()
        plt.savefig(output_dir + '/previous_' + str(graph_x) + '_NNconv.pdf', bbox_inches='tight')

    settings.finish_simulation(params.to_dict(), output_dir, output_png_path)
