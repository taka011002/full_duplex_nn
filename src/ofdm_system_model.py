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

    def __init__(self, block: int, subcarrier: int, CP: int, sigma: float, gamma=0.0, phi=0.0, PA_IBO_dB=5, PA_rho=2, LNA_alpha_1=5, LNA_alpha_2=2,
                 h_si_list=None,
                 h_s_list=None, h_si_len=1, h_s_len=1, receive_antenna=1, tx_iqi=True, pa=True, lna=True, rx_iqi=True):
        self.block = block
        self.subcarrier = subcarrier
        self.CP = CP
        self.subcarrier_CP = subcarrier + CP

        # 必要な行列を生成する
        dft_mat = dft(subcarrier, scale="sqrtn")
        self.dft_mat = dft_mat
        self.idft_mat = dft_mat.conj().T

        self.cp_zero = np.hstack((np.zeros((subcarrier, CP)), np.eye(subcarrier)))

        # ここまで

        # 送信信号
        self.d = np.random.choice([0, 1], (subcarrier * 2 * block, 1))
        x_n = m.modulate_qpsk(self.d)
        x_p = x_n.reshape((subcarrier, block), order='F')
        x_idft = np.matmul(self.idft_mat, x_p)
        x_cp = ofdm.add_cp(x_idft, CP)
        x = x_cp
        self.x = x_cp.flatten(order='F')
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
        s_p = s_n.reshape((subcarrier, block), order='F')
        s_idft = np.matmul(self.idft_mat, s_p)
        s_cp = ofdm.add_cp(s_idft, CP)
        self.tilde_s = s_cp

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

        self.y = np.zeros((self.subcarrier_CP * block, receive_antenna), dtype=complex)
        for receive_antenna_i in range(receive_antenna):
            h_si = h_si_list[receive_antenna_i]
            h_s = h_s_list[receive_antenna_i]

            toeplitz_h_si = ofdm.toeplitz_channel(h_si.T, h_si_len, subcarrier, CP)
            toeplitz_h_s = ofdm.toeplitz_channel(h_s.T, h_s_len, subcarrier, CP)

            r = np.matmul(toeplitz_h_si, x_rx) + np.matmul(toeplitz_h_s, s_rx) + m.awgn((subcarrier + CP, block), sigma)

            # 受信側非線形
            if lna == True:
                r = m.polynomial_amplifier(r, LNA_alpha_1, LNA_alpha_2)

            if rx_iqi == True:
                r = m.iq_imbalance(r, gamma, phi)

            y = r.flatten(order='F')

            self.y[:, receive_antenna_i] = y

    def demodulate_ofdm(self, y):
        one_block = self.subcarrier_CP
        y_p = y.reshape((one_block, -1), order='F')
        y_removed_cp = np.matmul(self.cp_zero, y_p)
        y_dft = np.matmul(self.dft_mat, y_removed_cp)
        s_s = y_dft.flatten(order='F')
        return s_s

    def demodulate_ofdm_dft(self, y, h_s, h_s_len):
        Hc = ofdm.circulant_channel(h_s.T, h_s_len, self.subcarrier)
        D = self.dft_mat @ Hc @ self.idft_mat
        D_1 = np.linalg.inv(D)

        one_block = self.subcarrier_CP
        y_p = y.reshape((one_block, -1), order='F')
        y_removed_cp = np.matmul(self.cp_zero, y_p)
        y_dft = np.matmul(self.dft_mat, y_removed_cp)
        s_s = np.matmul(D_1, y_dft)
        s_s = s_s.flatten(order='F')
        return s_s

    def demodulate_ofdm_dft_mmse(self, y, h_s, h_s_len, sigma):
        Hc = ofdm.circulant_channel(h_s.T, h_s_len, self.subcarrier)
        D = self.dft_mat @ Hc @ self.idft_mat
        MMSE = D.conj().T / (D*D.conj().T + sigma ** 2)
        D_1 = MMSE

        one_block = self.subcarrier_CP
        y_p = y.reshape((one_block, -1), order='F')
        y_removed_cp = np.matmul(self.cp_zero, y_p)
        y_dft = np.matmul(self.dft_mat, y_removed_cp)
        s_s = np.matmul(D_1, y_dft)
        s_s = s_s.flatten(order='F')
        return s_s