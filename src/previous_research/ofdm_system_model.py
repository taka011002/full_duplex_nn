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
        self.sigma = sigma
        self.gamma = gamma
        self.phi = phi
        self.PA_IBO_dB = PA_IBO_dB
        self.PA_rho = PA_rho
        self.LNA_IBO_dB = LNA_IBO_dB
        self.LNA_rho = LNA_rho
        self.h_si_list = h_si_list
        self.h_s_list = h_s_list
        self.h_si_len = h_si_len
        self.h_s_len = h_s_len
        self.receive_antenna = receive_antenna
        self.tx_iqi = tx_iqi
        self.pa = pa
        self.lna = lna
        self.rx_iqi = rx_iqi

        self.subcarrier_CP = subcarrier + CP
        self.dft_mat = dft(self.subcarrier, scale="sqrtn")
        self.idft_mat = self.dft_mat.conj().T
        self.cp_zero = np.hstack((np.zeros((self.subcarrier, self.CP)), np.eye(self.subcarrier)))

        Hc = ofdm.circulant_channel(self.h_s_list[0].T, self.h_s_len, self.subcarrier)
        D = self.dft_mat @ Hc @ self.idft_mat
        self.D_1 = np.linalg.inv(D)


    def transceive_s(self):
        # 送信信号
        self.d = np.random.choice([0, 1], (self.subcarrier * 2 * self.block, 1))
        x_n = m.modulate_qpsk(self.d)
        x_p = x_n.reshape((self.subcarrier, self.block), order='F')
        x_idft = np.matmul(self.idft_mat, x_p)
        x_cp = ofdm.add_cp(x_idft, self.CP)
        x = x_cp
        self.x = x_cp.flatten(order='F')
        tx_x = x

        # 送信側非線形
        if self.tx_iqi == True:
            tx_x = m.iq_imbalance(tx_x, self.gamma, self.phi)

        if self.pa == True:
            tx_x = m.sspa_rapp_ibo(tx_x, self.PA_IBO_dB, self.PA_rho, ofdm=True)

        x_rx = tx_x
        if self.h_si_len > 1:
            x_rx = np.zeros((self.h_si_len - 1 + tx_x.shape[0], tx_x.shape[1]), dtype=complex)
            x_rx[:(self.h_si_len - 1), 1:] = tx_x[-(self.h_si_len - 1):, :-1]
            x_rx[(self.h_si_len - 1):, :] = tx_x

        self.y = np.zeros((self.subcarrier_CP * self.block, self.receive_antenna), dtype=complex)
        for receive_antenna_i in range(self.receive_antenna):
            h_si = self.h_si_list[receive_antenna_i]

            toeplitz_h_si = ofdm.toeplitz_channel(h_si.T, self.h_si_len, self.subcarrier, self.CP)

            r = np.matmul(toeplitz_h_si, x_rx) + m.awgn((self.subcarrier + self.CP, self.block), self.sigma)

            # 受信側非線形
            if self.lna == True:
                ## TODO a_satがSIありとなしで変わらないようにする．
                r = m.sspa_rapp_ibo(r, self.LNA_IBO_dB, self.LNA_rho).squeeze()

            if self.rx_iqi == True:
                r = m.iq_imbalance(r, self.gamma, self.phi)

            y = r.flatten(order='F')

            self.y[:, receive_antenna_i] = y

    def transceive_si_s(self):
        # 送信信号
        self.d = np.random.choice([0, 1], (self.subcarrier * 2 * self.block, 1))
        x_n = m.modulate_qpsk(self.d)
        x_p = x_n.reshape((self.subcarrier, self.block), order='F')
        x_idft = np.matmul(self.idft_mat, x_p)
        x_cp = ofdm.add_cp(x_idft, self.CP)
        x = x_cp
        self.x = x_cp.flatten(order='F')
        tx_x = x

        # 送信側非線形
        if self.tx_iqi == True:
            tx_x = m.iq_imbalance(tx_x, self.gamma, self.phi)

        if self.pa == True:
            tx_x = m.sspa_rapp_ibo(tx_x, self.PA_IBO_dB, self.PA_rho, ofdm=True)

        # 希望信号
        self.d_s = np.random.choice([0, 1], (self.subcarrier * 2 * self.block, 1))
        s_n = m.modulate_qpsk(self.d_s)
        s_p = s_n.reshape((self.subcarrier, self.block), order='F')
        s_idft = np.matmul(self.idft_mat, s_p)
        s_cp = ofdm.add_cp(s_idft, self.CP)
        self.tilde_s = s_cp

        x_rx = tx_x
        if self.h_si_len > 1:
            x_rx = np.zeros((self.h_si_len - 1 + tx_x.shape[0], tx_x.shape[1]), dtype=complex)
            x_rx[:(self.h_si_len - 1), 1:] = tx_x[-(self.h_si_len - 1):, :-1]
            x_rx[(self.h_si_len - 1):, :] = tx_x

        s_rx = s_cp
        if self.h_s_len > 1:
            s_rx = np.zeros((self.h_s_len - 1 + s_cp.shape[0], s_cp.shape[1]), dtype=complex)
            s_rx[:(self.h_s_len - 1), 1:] = s_cp[-(self.h_s_len - 1):, :-1]
            s_rx[(self.h_s_len - 1):, :] = s_cp

        self.y = np.zeros((self.subcarrier_CP * self.block, self.receive_antenna), dtype=complex)
        for receive_antenna_i in range(self.receive_antenna):
            h_si = self.h_si_list[receive_antenna_i]
            h_s = self.h_s_list[receive_antenna_i]

            toeplitz_h_si = ofdm.toeplitz_channel(h_si.T, self.h_si_len, self.subcarrier, self.CP)
            toeplitz_h_s = ofdm.toeplitz_channel(h_s.T, self.h_s_len, self.subcarrier, self.CP)

            r = np.matmul(toeplitz_h_si, x_rx) + np.matmul(toeplitz_h_s, s_rx) + m.awgn((self.subcarrier + self.CP, self.block), self.sigma)

            # 受信側非線形
            if self.lna == True:
                ## TODO a_satがSIありとなしで変わらないようにする．
                r = m.sspa_rapp_ibo(r, self.LNA_IBO_dB, self.LNA_rho).squeeze()

            if self.rx_iqi == True:
                r = m.iq_imbalance(r, self.gamma, self.phi)

            y = r.flatten(order='F')

            self.y[:, receive_antenna_i] = y


    def demodulate_ofdm(self, y):
        one_block = self.subcarrier_CP
        y_p = y.reshape((one_block, -1), order='F')
        y_removed_cp = np.matmul(self.cp_zero, y_p)
        y_dft = np.matmul(self.dft_mat, y_removed_cp)
        s_s = np.matmul(self.D_1, y_dft)
        s_s = s_s.flatten(order='F')
        return s_s