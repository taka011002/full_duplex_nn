from src import modules as m
import numpy as np


class SystemModel:
    def __init__(self, n, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2, h_si=None, h_s=None, h_si_len=1, h_s_len=1):
        # 送信信号
        self.d = np.random.choice([0, 1], n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形
        self.x_iq = m.iq_imbalance(self.x, gamma, phi)
        self.x_pa = m.sspa_rapp_ibo(self.x_iq, PA_IBO_dB, PA_rho)

        # 通信路
        # 通信路がランダムの場合
        if h_si is None:
            h_si = m.channel(size=self.x_pa.size)
        if h_s is None:
            h_s = m.channel(size=self.s.size)

        self.h_si = h_si
        self.h_s = h_s

        self.y_si = h_si * np.reshape(np.array([self.x_pa[i:i+h_si_len] for i in range(self.x_pa.size-h_si_len+1)]), (self.x_pa.size-h_si_len+1, h_si_len))
        self.y_s = h_s * np.reshape(np.array([self.s[i:i+h_s_len] for i in range(self.s.size-h_s_len+1)]), (self.s.size-h_s_len+1, h_s_len))
        self.r = self.y_si + self.y_s + m.awgn(self.y_s.shape, sigma, h_s_len)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(self.r, LNA_IBO_dB, LNA_rho)