from src import modules as m
import numpy as np
from scipy.linalg import dft
from src import ofdm


class OFDMSystemModel:
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

    def __init__(self, block, subcarrier, CP, sigma, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_IBO_dB=5, LNA_rho=2,
                 h_si_list=None,
                 h_s_list=None, h_si_len=1, h_s_len=1, receive_antenna=1, tx_iqi=True, pa=True, lna=True, rx_iqi=True):
        # 必要な行列を生成する
        dft_mat = dft(subcarrier, scale="sqrtn")
        idft_mat = dft_mat.conj().T
        circulant_h_si = ofdm.circulant_channel(h_si_list, h_si_len, subcarrier)
        circulant_h_s = ofdm.circulant_channel(h_s_list, h_s_len, subcarrier)

        D_si = dft_mat @ circulant_h_si @ idft_mat
        D_s = dft_mat @ circulant_h_s @ idft_mat

        D_si_inv = np.linalg.inv(D_si)
        D_s_inv = np.linalg.inv(D_s)
        # ここまで

        # 送信信号
        self.d = np.random.choice([0, 1], (subcarrier * 2 * block, 1))
        x_n = m.modulate_qpsk(self.d)
        self.x = x_n # シリアルの状態を保持する
        x = x_n.reshape(subcarrier, block)
        x_idft = np.matmul(idft_mat, x)

        # 送信側非線形
        if tx_iqi == True:
            self.x_iq = m.iq_imbalance(x_idft, gamma, phi)
        else:
            self.x_iq = x_idft

        if pa == True:
            self.x_pa = m.sspa_rapp_ibo(self.x_iq, PA_IBO_dB, PA_rho, ofdm=True)
        else:
            self.x_pa = self.x_iq

        # 希望信号
        self.d_s = np.random.choice([0, 1], (subcarrier * 2 * block, 1))
        s_n = m.modulate_qpsk(self.d_s)
        self.s = s_n # シリアルの状態を保持する
        s = s_n.reshape(subcarrier, block)
        s_idft = np.matmul(idft_mat, s)

        r = np.matmul(circulant_h_si, x_idft) + np.matmul(circulant_h_s, s_idft) + m.awgn((subcarrier, block), sigma)

        # 受信側非線形
        if lna == True:
            y_lna = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho).squeeze()
        else:
            y_lna = r

        if rx_iqi == True:
            y_iq = m.iq_imbalance(y_lna, gamma, phi)
        else:
            y_iq = y_lna

        # CPは外し、DFT処理も行う
        y_dft = np.matmul(dft_mat, y_iq)
        y = y_dft.reshape(subcarrier * block, 1) # シリアルの状態を保持する

        self.y = y