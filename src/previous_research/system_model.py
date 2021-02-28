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
                 h_si: np.ndarray = None, h_si_len=1, h_s=None, h_s_len=1, tx_iqi=True, pa=True, lna=True, rx_iqi=True):
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
        self.tx_iqi = tx_iqi
        self.pa = pa
        self.lna = lna
        self.rx_iqi = rx_iqi

    def transceive_si(self, n):
        offset_n = n + 2 * (self.h_si_len - 1)  # 遅延を取る為に多く作っておく

        # 送信信号
        self.d = np.random.choice([0, 1], offset_n)
        self.x = m.modulate_qpsk(self.d)

        # 送信側非線形
        if self.tx_iqi == True:
            self.x_iq = m.iq_imbalance(self.x, self.gamma, self.phi)
        else:
            self.x_iq = self.x

        if self.pa == True:
            self.x_pa = m.sspa_rapp_ibo(self.x_iq, self.PA_IBO_dB, self.PA_rho)
        else:
            self.x_pa = self.x_iq

        # 通信路
        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_x_pa = np.array([self.x_pa[i:i + self.h_si_len] for i in range(self.x_pa.size - self.h_si_len + 1)])
        chanels_y_si = self.h_si * chanels_x_pa
        y_si = np.sum(chanels_y_si, axis=1)
        r = y_si + m.awgn(y_si.shape, self.sigma)

        # 受信側非線形
        if self.lna == True:
            y_lna = m.sspa_rapp_ibo(r)
        else:
            y_lna = r
        
        if self.rx_iqi == True:
            y_iq = m.iq_imbalance(y_lna, self.gamma, self.phi)
        else:
            y_iq = y_lna

        self.y = y_iq

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
        if self.lna == True:
            y_lna = m.polynomial_amplifier(r)
        else:
            y_lna = r
        
        if self.rx_iqi == True:
            y_iq = m.iq_imbalance(y_lna, self.gamma, self.phi)
        else:
            y_iq = y_lna

        self.y = y_iq

    def transceive_si_s(self, n):
        offset_n = n + 2 * (self.h_si_len - 1)  # 遅延を取る為に多く作っておく

        # 送信信号
        self.d = np.random.choice([0, 1], offset_n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], offset_n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形
        if self.tx_iqi == True:
            self.x_iq = m.iq_imbalance(self.x, self.gamma, self.phi)
        else:
            self.x_iq = self.x

        if self.pa == True:
            self.x_pa = m.sspa_rapp_ibo(self.x_iq, self.PA_IBO_dB, self.PA_rho)
        else:
            self.x_pa = self.x_iq

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
        if self.lna == True:
            y_lna = m.polynomial_amplifier(r)
        else:
            y_lna = r
        
        if self.rx_iqi == True:
            y_iq = m.iq_imbalance(y_lna, self.gamma, self.phi)
        else:
            y_iq = y_lna

        self.y = y_iq
        
    def set_lna_a_sat(self, n, LNA_IBO_dB):
        # TODO 調整する
        offset_n = n + 2 * (self.h_si_len - 1)  # 遅延を取る為に多く作っておく

        # 送信信号
        d = np.random.choice([0, 1], offset_n)
        x = m.modulate_qpsk(d)

        # 希望信号
        d_s = np.random.choice([0, 1], offset_n)
        s = m.modulate_qpsk(d_s)

        # 送信側非線形
        if self.tx_iqi == True:
            x_iq = m.iq_imbalance(x, self.gamma, self.phi)
        else:
            x_iq = x

        if self.pa == True:
            x_pa = m.sspa_rapp_ibo(x_iq, self.PA_IBO_dB, self.PA_rho)
        else:
            x_pa = x_iq

        # 通信路
        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_x_pa = np.array([x_pa[i:i + self.h_si_len] for i in range(x_pa.size - self.h_si_len + 1)])
        chanels_y_si = self.h_si * chanels_x_pa
        y_si = np.sum(chanels_y_si, axis=1)

        # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
        chanels_s = np.array([s[i:i + self.h_s_len] for i in range(s.size - self.h_s_len + 1)])
        chanels_s = self.h_s * chanels_s
        y_s = np.sum(chanels_s, axis=1)

        r = y_si + y_s + m.awgn(y_si.shape, self.sigma)
        self.a_sat = m.a_sat(r, LNA_IBO_dB)
