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

    def __init__(self, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2,
                 h_si: np.ndarray = None, h_si_len=1, h_s=None, h_s_len=1):
        self.sigma = sigma
        self.gamma = gamma
        self.phi = phi
        self.PA_IBO_dB = PA_IBO_dB
        self.PA_rho = PA_rho
        self.LNA_IBO_dB = LNA_IBO_dB
        self.LNA_rho = LNA_rho
        self.h_si = h_si
        self.h_si_len = h_si_len
        self.h_s = h_s
        self.h_s_len = h_s_len

    def transceive_si(self, n):
        offset_n = n + 2 * (self.h_si_len - 1)  # 遅延を取る為に多く作っておく

        # 送信信号
        self.d = np.random.choice([0, 1], offset_n)
        self.x = m.modulate_qpsk(self.d)

        # 送信側非線形
        self.x_iq = m.iq_imbalance(self.x, self.gamma, self.phi)
        self.x_pa = m.sspa_rapp_ibo(self.x_iq, self.PA_IBO_dB, self.PA_rho)

        # 通信路
        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_x_pa = np.array([self.x_pa[i:i + self.h_si_len] for i in range(self.x_pa.size - self.h_si_len + 1)])
        chanels_y_si = self.h_si * chanels_x_pa
        y_si = np.sum(chanels_y_si, axis=1)
        r = y_si + m.awgn(y_si.shape, self.sigma)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, self.LNA_IBO_dB, self.LNA_rho).squeeze()

    def transceive_s(self, n):
        offset_n = n + 2 * (self.h_s_len - 1)  # 遅延を取る為に多く作っておく

        # 希望信号
        self.d_s = np.random.choice([0, 1], offset_n)
        self.s = m.modulate_qpsk(self.d_s)

        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_s = np.array([self.s[i:i + self.h_s_len] for i in range(self.s.size - self.h_s_len + 1)])
        chanels_s = self.h_s * chanels_s
        y_s = np.sum(chanels_s, axis=1)

        r = y_s + m.awgn(y_s.shape, self.sigma)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, self.LNA_IBO_dB, self.LNA_rho).squeeze()

    def transceive_si_s(self, n):
        offset_n = n + 2 * (self.h_si_len - 1)  # 遅延を取る為に多く作っておく

        # 送信信号
        self.d = np.random.choice([0, 1], offset_n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], offset_n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形
        self.x_iq = m.iq_imbalance(self.x, self.gamma, self.phi)
        self.x_pa = m.sspa_rapp_ibo(self.x_iq, self.PA_IBO_dB, self.PA_rho)

        # 通信路
        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_x_pa = np.array([self.x_pa[i:i + self.h_si_len] for i in range(self.x_pa.size - self.h_si_len + 1)])
        chanels_y_si = self.h_si * chanels_x_pa
        y_si = np.sum(chanels_y_si, axis=1)

        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_s = np.array([self.s[i:i + self.h_s_len] for i in range(self.s.size - self.h_s_len + 1)])
        chanels_s = self.h_s * chanels_s
        y_s = np.sum(chanels_s, axis=1)

        r = y_si + y_s + m.awgn(y_si.shape, self.sigma)

        # 受信側非線形
        self.y = m.sspa_rapp_ibo(r, self.LNA_IBO_dB, self.LNA_rho).squeeze()
