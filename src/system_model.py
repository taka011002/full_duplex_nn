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

    def __init__(self, n, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2, h_si_list=None,
                 h_s_list=None, h_si_len=1, h_s_len=1, receive_antenna=1, tx_iqi=True, pa=True, lna=True, rx_iqi=True):
        # 送信信号
        self.d = np.random.choice([0, 1], n)
        self.x = m.modulate_qpsk(self.d)

        # 希望信号
        self.d_s = np.random.choice([0, 1], n)
        self.s = m.modulate_qpsk(self.d_s)

        # 送信側非線形

        if tx_iqi == True:
            self.x_iq = m.iq_imbalance(self.x, gamma, phi)
        else:
            self.x_iq = self.x
        
        if pa == True:
            self.x_pa = m.sspa_rapp_ibo(self.x_iq, PA_IBO_dB, PA_rho)
        else:
            self.x_pa = self.x_iq

        # 通信路
        # 通信路がランダムの場合
        if h_si_list is None:
            h_si_list = [m.channel(size=self.x_pa.size)]
        if h_s_list is None:
            h_s_list = [m.channel(size=self.s.size)]

        self.h_si_list = h_si_list
        self.h_s_list = h_s_list

        x_len = self.x_pa.size - h_si_len + 1 # 周波数選択性の場合，時間のずれを考慮すると長さがnではなくなる
        self.y = np.zeros((x_len, receive_antenna), dtype=complex)
        for i, (h_si, h_s) in enumerate(zip(h_si_list, h_s_list)):
            chanels_x_pa = np.array([self.x_pa[i:i + h_si_len] for i in range(self.x_pa.size - h_si_len + 1)]) # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
            chanels_y_si = h_si * chanels_x_pa
            y_si = np.sum(chanels_y_si, axis=1)
            chanels_s = np.array([self.s[i:i + h_s_len] for i in range(self.s.size - h_s_len + 1)]) # [[x[n], x[n-1]], x[x-1], x[n-1]]のように通信路の数に合わせる
            chanels_s = h_s * chanels_s
            y_s = np.sum(chanels_s, axis=1)
            r = y_si + y_s + m.awgn(y_s.shape, sigma)

            # 受信側非線形
            if lna == True:
                y_lna = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho).squeeze()
            else:
                y_lna = r

            if rx_iqi == True:
                y_iq = m.iq_imbalance(y_lna, gamma, phi)
            else:
                y_iq = y_lna            

            self.y[:, i] = y_iq