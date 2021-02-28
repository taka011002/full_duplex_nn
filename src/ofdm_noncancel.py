from src.ofdm_system_model import OFDMSystemModel
from src import modules as m
import numpy as np


def simulation(block: int, subcarrier: int, CP: int, sigma: float, gamma: float,
               phi: float, PA_IBO_dB: float, PA_rho: float, LNA_IBO_dB: float, LNA_rho: float,
               h_si_list: list, h_s_list: list, h_si_len: int, h_s_len: int, TX_IQI: bool, PA: bool, LNA: bool, RX_IQI: bool,
               trainingRatio: float, compensate_iqi: bool=False, receive_antenna=1, equalizer='ZF') -> np.ndarray:
    h_si = h_si_list[0:receive_antenna]
    h_s = h_s_list[0:receive_antenna]

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
        h_si,
        h_s,
        h_si_len,
        h_s_len,
        receive_antenna,
        TX_IQI,
        PA,
        LNA,
        RX_IQI
    )

    s_hat_array = np.zeros((receive_antenna, system_model.d.shape[0] // 2), dtype=complex)
    for i in range(receive_antenna):
        y = system_model.y[:, i]
        if compensate_iqi is True:
            y = m.compensate_iqi(y.flatten(order='F'), gamma, phi)
        if equalizer == 'MMSE':
            s_hat_array[i] = system_model.demodulate_ofdm_dft_mmse(y, h_s_list[i], h_s_len, sigma)
        else:
            # MMSE以外が指定されている場合は，ZF
            s_hat_array[i] = system_model.demodulate_ofdm_dft(y, h_s_list[i], h_s_len)

    s_hat = np.sum(s_hat_array, axis=0)
    d_s_hat = m.demodulate_qpsk(s_hat)

    error = np.sum(d_s_hat != system_model.d_s.flatten())
    return error