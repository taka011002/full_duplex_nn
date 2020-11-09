from src.system_model import SystemModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 7,
        'PA_rho': 2,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,
    }

    system_model = SystemModel(
        params['n'],
        0,
        params['gamma'],
        params['phi'],
        params['PA_IBO_dB'],
        params['PA_rho'],
        params['LNA_IBO_dB'],
        params['LNA_rho'],
    )


    plots = 10
    plt.figure()
    plt.scatter(system_model.r[::plots].real, system_model.r[::plots].imag, color="black", label="x")
    plt.scatter(system_model.y[::plots].real, system_model.y[::plots].imag, color="blue", label="x_iq")
    # plt.scatter(system_model.x_pa[::plots].real, system_model.x_pa[::plots].imag, color="red", label="x_pa")
    # plt.scatter(system_model.x_pa[::plots].real, system_model.x_pa[::plots].imag, color="red", label="x_pa")
    plt.legend()
    plt.show()
    print("end")