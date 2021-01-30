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
    SNR: int
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
    optimizer: str
    learningRate: float
    momentum: float
    trainingRatio: float
    batchSize: int
    delay: int
    standardization: bool
    seed: bool
    train_bits: int
    test_bits: int
    previous_test_bits: int

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


if __name__ == '__main__':
    SIMULATIONS_NAME = os.path.basename(__file__).split('.')[0]

    params_dict, output_dir = settings.init_simulation(SIMULATIONS_NAME, ofdm=True)
    params = Params.from_params_dict(params_dict)

    graph_x_array = np.linspace(params.graph_x_min, params.graph_x_max, params.graph_x_num, dtype=int)

    sigma = m.sigmas(params.SNR)
    proposal_sigma = sigma * np.sqrt(params.receive_antenna)  # 提案法のみ複数アンテナ

    errors = np.zeros((params.graph_x_num, params.trials))

    for trials_index in tqdm(range(params.trials)):
        h_si = []
        h_s = []

        for i in range(params.receive_antenna):
            h_si.append(m.exponential_decay_channel(1, params.h_si_len))
            h_s.append(m.exponential_decay_channel(1, params.h_s_len))

        for graph_x_index, nEpochs in enumerate(graph_x_array):
            nn_model = simulation(
                params.block,
                params.subcarrier,
                params.CP,
                proposal_sigma,
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
                nEpochs,
                params.batchSize,
                params.delay,
                params.standardization
            )

            errors[graph_x_index][trials_index] = nn_model.error

    result = Result(params, errors)
    with open(output_dir + '/result.pkl', 'wb') as f:
        pickle.dump(result, f)

    ber_fig, ber_ax = graph.new_ber_canvas("Training Epoch", params.graph_x_min, params.graph_x_max)

    n_sum = params.test_bits * params.trials
    errors_sum = np.sum(errors, axis=1)
    bers = errors_sum / n_sum
    ber_ax.plot(graph_x_array, bers, color="k", marker='o', linestyle='--', label="Proposal")

    plt.tight_layout()
    # ber_ax.legend()
    plt.savefig(output_dir + '/SNR_BER.pdf', bbox_inches='tight')

    # slack通知用
    output_png_path = output_dir + '/SNR_BER.png'
    plt.savefig(output_png_path, bbox_inches='tight')

    settings.finish_simulation(params.to_dict(), output_dir, output_png_path)
