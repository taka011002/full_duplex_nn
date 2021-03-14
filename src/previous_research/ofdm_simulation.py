from src.previous_research.ofdm_system_model import OFDMSystemModel as PreviousOFDMSystemModel
from src.previous_research.ofdm_nn import OFDMNNModel as PreviousOFDMNNModel
from src import modules as m
import numpy as np
import tensorflow.keras as keras

def simulation(block: int, subcarrier: int, CP: int, sigma: float, gamma: float,
               phi: float, PA_IBO_dB: float, PA_rho: float, LNA_alpha_1: float, LNA_alpha_2: float,
               h_si_list: list, h_s_list: list, h_si_len: int, h_s_len: int ,TX_IQI: bool, PA: bool, LNA: bool,
               RX_IQI: bool, n_hidden: list, optimizer_key: str, learning_rate: float, momentum: float,
               trainingRatio: float, nEpochs: int, batchSize: int, compensate_iqi: bool=False, receive_antenna=1, equalizer='ZF') -> PreviousOFDMNNModel:
    keras.backend.clear_session() # 複数試行行うとメモリリークするのでその対策

    h_si = h_si_list[0:receive_antenna]
    h_s = h_s_list[0:receive_antenna]

    training_blocks = int(block * trainingRatio)
    test_blocks = block - training_blocks

    train_system_model = PreviousOFDMSystemModel(
        training_blocks,
        subcarrier,
        CP,
        sigma,
        gamma,
        phi,
        PA_IBO_dB,
        PA_rho,
        LNA_alpha_1,
        LNA_alpha_2,
        h_si,
        h_s,
        h_si_len,
        h_s_len,
        receive_antenna,
        TX_IQI,
        PA,
        LNA,
        RX_IQI,
    )

    test_system_model = PreviousOFDMSystemModel(
        test_blocks,
        subcarrier,
        CP,
        sigma,
        gamma,
        phi,
        PA_IBO_dB,
        PA_rho,
        LNA_alpha_1,
        LNA_alpha_2,
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

    train_system_model.transceive_s()
    test_system_model.transceive_s()

    nn_model_list = []
    for i in range(receive_antenna):
        nn_model = PreviousOFDMNNModel(
            n_hidden,
            optimizer_key,
            learning_rate,
            h_si_len,
            momentum
        )

        nn_model.learn(
            train_system_model.x,
            train_system_model.y[:, i],
            test_system_model.x,
            test_system_model.y[:, i],
            nEpochs,
            batchSize,
            h_si_len,
        )

        nn_model_list.append(nn_model)

    pred_system_model = PreviousOFDMSystemModel(
        test_blocks,
        subcarrier,
        CP,
        sigma,
        gamma,
        phi,
        PA_IBO_dB,
        PA_rho,
        LNA_alpha_1,
        LNA_alpha_2,
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
    pred_system_model.transceive_si_s()

    s_hat_array = np.zeros((receive_antenna, pred_system_model.d.shape[0] // 2), dtype=complex)
    for i in range(receive_antenna):
        nn_model = nn_model_list[i]
        nn_model.cancel(pred_system_model.x,  h_si_len)
        cancelled_y = pred_system_model.y[:, i].reshape(1, -1) - nn_model.y_hat

        if compensate_iqi is True:
            cancelled_y = m.compensate_iqi(cancelled_y.flatten(order='F'), gamma, phi)

        if equalizer == 'MMSE':
            s_hat_array[i] = pred_system_model.demodulate_ofdm_mmse(cancelled_y, h_s[i], sigma)
        else:
            # MMSE以外が指定されている場合は，ZF
            s_hat_array[i] = pred_system_model.demodulate_ofdm(cancelled_y, h_s[i])


    # s_hat_array = s_hat_array[0, :].reshape(1, -1)
    # s_hat_array = s_hat_array[1, :].reshape(1, -1)
    # s_hat_array = s_hat_array[2, :].reshape(1, -1)

    # 推定信号をデータへ復調する
    s_hat = np.sum(s_hat_array, axis=0)
    # s_hat = s_hat / receive_antenna
    d_s_hat = m.demodulate_qpsk(s_hat)

    # 元々の外部信号のデータ
    d_s_test = pred_system_model.d_s[:d_s_hat.size].flatten()
    error = np.sum(d_s_test != d_s_hat)
    nn_model.error = error

    return nn_model
