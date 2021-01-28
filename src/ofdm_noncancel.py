from src.ofdm_system_model import OFDMSystemModel
from src import modules as m
import numpy as np


def simulation(block: int, subcarrier: int, CP: int, sigma: float, gamma: float,
               phi: float, PA_IBO_dB: float, PA_rho: float, LNA_IBO_dB: float, LNA_rho: float,
               h_si_list: list, h_s_list: list, h_si_len: int, h_s_len: int, TX_IQI: bool, PA: bool, LNA: bool, RX_IQI: bool,
               trainingRatio: float) -> np.ndarray:
    ## 受信アンテナ数は1本のみで動作
    receive_antenna = 1
    h_si = []
    h_si.append(h_si_list[0])
    h_s = []
    h_s.append(h_s_list[0])

    training_block = int(block * trainingRatio)

    system_model = OFDMSystemModel(
        training_block,
        subcarrier,
        CP,
        sigma,
        gamma,
        phi,
        PA_IBO_dB,
        PA_rho,
        LNA_IBO_dB,
        LNA_rho,
        h_si_list,
        h_s_list,
        h_si_len,
        h_s_len,
        receive_antenna,
        TX_IQI,
        PA,
        LNA,
        RX_IQI
    )

    s_hat = system_model.demodulate_ofdm_dft(system_model.y, h_s_list[0], h_s_len)
    d_s_hat = m.demodulate_qpsk(s_hat)

    error = np.sum(d_s_hat != system_model.d_s.flatten())
    return error