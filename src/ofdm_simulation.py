from src.ofdm_system_model import OFDMSystemModel
from src.ofdm_nn import OFDMNNModel


def simulation(block: int, subcarrier: int, CP: int, sigma: float, gamma: float,
               phi: float, PA_IBO_dB: float, PA_rho: float, LNA_IBO_dB: float, LNA_rho: float,
               h_si_list: list, h_s_list: list, h_si_len: int, h_s_len: int, receive_antenna: int,
               TX_IQI: bool, PA: bool, LNA: bool, RX_IQI: bool, n_hidden: list, optimizer_key: str,
               learning_rate: float, momentum: float, trainingRatio: float, nEpochs: int, batchSize: int,
               delay: int, standardization: bool) -> OFDMNNModel:
    training_block = int(block * trainingRatio)
    test_block = block - training_block

    train_system_model = OFDMSystemModel(
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

    test_system_model = OFDMSystemModel(
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

    nn_model = OFDMNNModel(
        n_hidden,
        optimizer_key,
        learning_rate,
        h_si_len,
        h_s_len,
        receive_antenna,
        momentum
    )

    nn_model.learn(
        train_system_model,
        test_system_model,
        trainingRatio,
        nEpochs,
        batchSize,
        h_si_len,
        h_s_len,
        receive_antenna,
        delay,
        standardization
    )

    return nn_model
