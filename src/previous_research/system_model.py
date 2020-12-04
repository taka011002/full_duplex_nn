from src import modules as m
import numpy as np


class SystemModel:
    d: np.ndarray
    x: np.ndarray
    d_s: np.ndarray
    s: np.ndarray
    x_iq: np.ndarray
    x_pa: np.ndarray
    h_si: np.ndarray
    h_s: np.ndarray
    y_si: np.ndarray
    y_s: np.ndarray
    r: np.ndarray
    y: np.ndarray

    def learning_phase(self, n, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2, h_si: np.ndarray=None,
                 h_si_len=1):
        # 送信信号
        self.d = np.random.choice([0, 1], n)
        self.x = m.modulate_qpsk(self.d)

        # 送信側非線形
        self.x_iq = m.iq_imbalance(self.x, gamma, phi)
        self.x_pa = m.sspa_rapp_ibo(self.x_iq, PA_IBO_dB, PA_rho)

        # 通信路
        self.h_si = h_si
        chanels_x_pa = np.array([self.x_pa[i:i + h_si_len] for i in range(self.x_pa.size - h_si_len + 1)]) # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_y_si = h_si * chanels_x_pa
        y_si = np.sum(chanels_y_si, axis=1)
        r = y_si + m.awgn(y_si.shape, sigma, h_si_len)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho).squeeze()