from src import modules as m
from src.system_model import SystemModel
import numpy as np

if __name__ == '__main__':
    params = {
        'n': 2 * 10 ** 4,  # サンプルのn数
        'gamma': 0.3,
        'phi': 3.0,
        'PA_IBO_dB': 3,
        'PA_rho': 2,

        'LNA_IBO_dB': 3,
        'LNA_rho': 2,

        'SNR_MIN': 0,
        'SNR_MAX': 12,
        'sur_num': 12,
    }

    snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['sur_num'])
    sigmas = m.sigmas(snrs_db)  # SNR(dB)を元に雑音電力を導出

    for index, sigma in enumerate(sigmas):
        system_model = SystemModel(
            params['n'],
            sigma,
            params['gamma'],
            params['phi'],
            params['PA_IBO_dB'],
            params['PA_rho'],
            params['LNA_IBO_dB'],
            params['LNA_rho'],
        )
