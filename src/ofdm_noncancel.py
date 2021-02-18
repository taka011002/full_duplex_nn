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
    test_block = block - training_block

    system_model = OFDMSystemModel(
        test_block,
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

    y = m.compensate_iqi(system_model.y.flatten(order='F'), gamma, phi)
    s_hat = system_model.demodulate_ofdm_dft(y, h_s_list[0], h_s_len)
    compensate_iqi = m.compensate_iqi(s_hat, gamma, phi)
    d_s_hat = m.demodulate_qpsk(compensate_iqi)

    error = np.sum(d_s_hat != system_model.d_s.flatten())
    return error