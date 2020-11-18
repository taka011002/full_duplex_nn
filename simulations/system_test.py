from src import modules as m
from src.system_model import SystemModel
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 1000,
        'PA_rho': 1000,

        'LNA_IBO_dB': 7,
        'LNA_rho': 2,

        'h_si_len': 1,
        'h_s_len': 1,
    }

    h_si = m.channel(1, params['h_si_len'])
    h_s = m.channel(1, params['h_s_len'])

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


    plt.figure()
    plt.scatter(system_model.y.real, system_model.y.imag, color="g", label="y")
    # plt.scatter(system_model.y_s[::plots].real, system_model.r[::plots].imag, color="blue", label="r")
    # plt.scatter(system_model.r.real, system_model.r.imag, color="blue", label="r")
    hensa = np.sqrt(np.var(system_model.y))
    normalize_y = system_model.y / hensa
    plt.scatter(normalize_y.real, normalize_y.imag, color="b", label="y_n_1")
    # sspa = m.sspa_rapp(system_model.r, 10000, 10000)
    # plt.scatter(sspa.real, sspa.imag, color="black", label="rapp")
    # plt.scatter(system_model.x_pa[::plots].real, system_model.x_pa[::plots].imag, color="red", label="x_pa")
    plt.scatter(system_model.x.real, system_model.x.imag, color="red", label="x")
    plt.legend()
    plt.show()
    print("end")