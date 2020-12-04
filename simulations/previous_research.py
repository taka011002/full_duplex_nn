from src import modules as m
from src.previous_research.nn import NNModel as PreviousNNModel
from src.previous_research.system_model import SystemModel as PreviousSystemModel
import matplotlib.pyplot as plt


if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research'

    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,

        'phi': 3.0,
        'PA_IBO_dB': 7,
        'PA_rho': 2,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,

        "h_si_len": 13,
        "n_hidden": 17,
        "learning_rate": 0.004,
        "batch_size": 32
    }

    h_si = m.channel(1, params['h_si_len'])
    system_model = PreviousSystemModel()
    system_model.learning_phase(
        params['n'],
        0,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
        h_si,
        params['h_si_len'],
    )

    plt.figure()
    plt.scatter(system_model.y.real, system_model.y.imag, color="g", label="y")
    plt.scatter(system_model.x.real, system_model.x.imag, color="red", label="x")
    plt.legend()
    plt.show()

    # previous_nn_model = PreviousNNModel(13, 17, 0.004)
