from src import modules as m
import numpy as np
from src import ofdm
from scipy.linalg import dft

#
# c = m.channel(1, 2, 1)
# ec = m.exponential_decay_channel(1, 1, 1, 0.23, 1)
#
# print(np.abs(ec))
# print(c)
# print(ec)

# 送信信号
# n = 100
N = 6 # サブキャリア数
P = 2 # CP長
M = 1 # 通信路長

# s = np.random.choice([0, 1], n * N)
# x = m.modulate_qpsk(s)

F = dft(N)
FH = F.conj().T

# h_si = m.channel(1, M).reshape(M)
s = np.random.choice([0, 1], (N * 2, 1))
x = m.modulate_qpsk(s)
x = x * np.sqrt(2) # 電力制限を一旦なくす
# x_idft = ofdm.IDFT(x)
x_idft = FH @ x
x_cp = ofdm.add_cp(x_idft, P)
# r = np.convolve(h_si, x_cp)
# r_remove_cp = ofdm.remove_cp(r, P, N)
# s = ofdm.DFT(r_remove_cp)

h_si = m.channel(1, M+1)
# h_si = np.array([1 + 1j, 1 + 1j]).reshape((1, 2))

H = ofdm.toeplitz_channel(h_si.T, M, N, P)
delay = np.zeros((M, 1))
x_hat = np.vstack((delay, x_cp))
r = H @ x_hat
# r = H[:, 1:] @ x_cp
Rcp = np.hstack((np.zeros((N, P)), np.eye(N, N)))
r_remove_cp = Rcp @ r

# Hcを作って計算する
Hc = ofdm.circulant_channel(h_si.T, M, N)
r_perfect = Hc @ x_idft
y = F @ r_remove_cp
D = F @ Hc @ FH
D_1 = np.linalg.inv(D)
x_hat = D_1 @ y
s_hat = m.demodulate_qpsk(x_hat.squeeze()).reshape((N * 2, 1))
error = np.sum(s != s_hat)
# p_error = np.sum(s != s_hat)


print(error)
# print(p_error)