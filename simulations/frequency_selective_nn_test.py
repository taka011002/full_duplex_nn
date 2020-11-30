from src.system_model import SystemModel
import matplotlib.pyplot as plt
from src.nn import NNModel
from src import modules as m

if __name__ == '__main__':
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 7,
        'PA_rho': 2,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,

        'SNR_MIN': 0,
        'SNR_MAX': 25,
        'SNR_NUM': 6,
        'SNR_AVERAGE': 50,

        'h_si_len': 2,
        'h_s_len': 2,

        'nHidden': 15,
        'nEpochs': 20,
        'learningRate': 0.004,
        'trainingRatio': 0.8,  # 全体のデータ数に対するトレーニングデータの割合
        'batchSize': 32,
    }

    h_si = m.channel()
    h_s = m.channel()

    system_model = SystemModel(
        params['n'],
        0,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
        h_si,
        h_s,
        params['h_si_len'],
        params['h_s_len'],
    )

    model = NNModel(
        params['nHidden'],
        params['learningRate'],
        params['h_si_len'],
        params['h_s_len'],
    )
    model.learn(
        system_model,
        params['trainingRatio'],
        params['nEpochs'],
        params['batchSize'],
        params['h_si_len'],
        params['h_s_len'],
    )

    # plots = 10
    # plt.figure()
    # plt.scatter(system_model.r[::plots].real, system_model.r[::plots].imag, color="black", label="x")
    # plt.scatter(system_model.y[::plots].real, system_model.y[::plots].imag, color="blue", label="x_iq")
    # plt.scatter(system_model.x_pa[::plots].real, system_model.x_pa[::plots].imag, color="red", label="x_pa")
    # plt.scatter(system_model.x_pa[::plots].real, system_model.x_pa[::plots].imag, color="red", label="x_pa")
    # plt.legend()
    # plt.show()
    print("end")