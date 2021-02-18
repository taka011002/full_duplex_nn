from src.previous_research.ofdm_system_model import OFDMSystemModel as PreviousOFDMSystemModel
from src.previous_research.ofdm_nn import OFDMNNModel as PreviousOFDMNNModel


def simulation(block: int, subcarrier: int, CP: int, sigma: float, gamma: float,
               phi: float, PA_IBO_dB: float, PA_rho: float, LNA_IBO_dB: float, LNA_rho: float,
               h_si_list: list, h_s_list: list, h_si_len: int, h_s_len: int ,TX_IQI: bool, PA: bool, LNA: bool,
               RX_IQI: bool, n_hidden: list, optimizer_key: str, learning_rate: float, momentum: float,
               trainingRatio: float, nEpochs: int, batchSize: int, compensate_iqi: bool=False) -> PreviousOFDMNNModel:
    ## 受信アンテナ数は1本のみで動作
    receive_antenna = 1
    h_si = []
    h_si.append(h_si_list[0])
    h_s = []
    h_s.append(h_s_list[0])

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

    test_system_model = PreviousOFDMSystemModel(
        test_blocks,
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

    train_system_model.transceive_s()
    test_system_model.transceive_s()

    nn_model = PreviousOFDMNNModel(
        n_hidden,
        optimizer_key,
        learning_rate,
        h_si_len,
        momentum
    )

    nn_model.learn(
        train_system_model,
        test_system_model,
        nEpochs,
        batchSize,
        h_si_len,
    )

    pred_system_model = PreviousOFDMSystemModel(
        test_blocks,
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
    pred_system_model.transceive_si_s()

    nn_model.cancel(pred_system_model,  h_si_len, compensate_iqi)

    return nn_model
