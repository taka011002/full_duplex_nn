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
        self.block = block
        self.subcarrier = subcarrier
        self.CP = CP
        self.subcarrier_CP = subcarrier + CP

        # 必要な行列を生成する
        dft_mat = dft(subcarrier, scale="sqrtn")
        self.dft_mat = dft_mat
        idft_mat = dft_mat.conj().T

        self.cp_zero = np.hstack((np.zeros((subcarrier, CP)), np.eye(subcarrier)))
        # self.circulant_h_si = ofdm.circulant_channel(h_si_list.T, h_si_len, subcarrier)
        circulant_h_s = ofdm.circulant_channel(h_s_list.T, h_s_len, subcarrier)
        D_s = dft_mat @ circulant_h_s @ idft_mat
        self.D_s_inv = np.linalg.inv(D_s)
        toeplitz_h_si = ofdm.toeplitz_channel(h_si_list.T, h_si_len, subcarrier, CP)
        toeplitz_h_s = ofdm.toeplitz_channel(h_s_list.T, h_s_len, subcarrier, CP)

        # ここまで

        # 送信信号
        self.d = np.random.choice([0, 1], (subcarrier * 2 * block, 1))
        x_n = m.modulate_qpsk(self.d)
        x_p = x_n.reshape(subcarrier, block)
        x_idft = np.matmul(idft_mat, x_p)
        x_cp = ofdm.add_cp(x_idft, CP)
        x = x_cp
        self.x = x_cp.flatten()
        tx_x = x

        # 送信側非線形
        if tx_iqi == True:
            tx_x = m.iq_imbalance(tx_x, gamma, phi)

        if pa == True:
            tx_x = m.sspa_rapp_ibo(tx_x, PA_IBO_dB, PA_rho, ofdm=True)

        # 希望信号
        self.d_s = np.random.choice([0, 1], (subcarrier * 2 * block, 1))
        s_n = m.modulate_qpsk(self.d_s)
        # self.s = s_n # シリアルの状態を保持する
        s_p = s_n.reshape(subcarrier, block)
        s_idft = np.matmul(idft_mat, s_p)
        s_cp = ofdm.add_cp(s_idft, CP)

        x_rx = tx_x
        if h_si_len > 1:
            x_rx = np.zeros((h_si_len - 1 + tx_x.shape[0], tx_x.shape[1]), dtype=complex)
            x_rx[:(h_si_len - 1), 1:] = tx_x[-(h_si_len - 1):, :-1]
            x_rx[(h_si_len - 1):, :] = tx_x

        s_rx = s_cp
        if h_s_len > 1:
            s_rx = np.zeros((h_s_len - 1 + s_cp.shape[0], s_cp.shape[1]), dtype=complex)
            s_rx[:(h_s_len - 1), 1:] = s_cp[-(h_s_len - 1):, :-1]
            s_rx[(h_s_len - 1):, :] = s_cp
        self.s_hs_rx = np.matmul(toeplitz_h_s, s_rx)

        r = np.matmul(toeplitz_h_si, x_rx) + self.s_hs_rx + m.awgn((subcarrier + CP, block), sigma)

        # 受信側非線形
        if lna == True:
            r = m.sspa_rapp_ibo(r, LNA_IBO_dB, LNA_rho).squeeze()

        if rx_iqi == True:
            r = m.iq_imbalance(r, gamma, phi)

        y = r.flatten()

        self.y = y

    def demodulate_ofdm(self, y):
        one_block = self.subcarrier_CP
        y_p = y.reshape((one_block, -1))
        y_removed_cp = np.matmul(self.cp_zero, y_p)
        y_dft = np.matmul(self.dft_mat, y_removed_cp)
        s_hat = np.matmul(self.D_s_inv, y_dft)
        s_s = s_hat.flatten()
        return s_s