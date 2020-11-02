import modules as m
import numpy as np


class SystemModel:
    def __init__(self, n, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2):
        # パラメータ
        self.gamma = gamma
        self.phi = phi

        # 送信信号
        self.d = np.random.choice([0, 1], n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形
        x_iq = m.iq_imbalance(self.x, self.gamma, self.phi)
        x_pa = m.sspa_rapp_ibo(x_iq, PA_IBO_dB, PA_rho)

        # 通信路
        y_si = m.channel(x_pa)
        y_o = m.channel(self.s)
        r = y_si + y_o + m.awgn(y_si.size, sigma)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho)