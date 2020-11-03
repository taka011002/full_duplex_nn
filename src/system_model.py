from src import modules as m
import numpy as np


class SystemModel:
    def __init__(self, n, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2, h_si=None, h_s=None):
        # 送信信号
        self.d = np.random.choice([0, 1], n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形
        x_iq = m.iq_imbalance(self.x, gamma, phi)
        x_pa = m.sspa_rapp_ibo(x_iq, PA_IBO_dB, PA_rho)

        # 通信路
        # 通信路がランダムの場合
        if h_si is None:
            h_si = m.channel(size=x_pa.size)
        if h_s is None:
            h_s = m.channel(size=self.s.size)

        y_si = x_pa * h_si
        y_s = self.s * h_s
        r = y_si + y_s + m.awgn(y_si.size, sigma)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho)