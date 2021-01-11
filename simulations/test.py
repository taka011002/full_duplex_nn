from src import modules as m
import numpy as np
from src import ofdm

#
# c = m.channel(1, 2, 1)
# ec = m.exponential_decay_channel(1, 1, 1, 0.23, 1)
#
# print(np.abs(ec))
# print(c)
# print(ec)

# 送信信号
# n = 100
N = 32 # サブキャリア数
P = 3 # CP長

# s = np.random.choice([0, 1], n * N)
# x = m.modulate_qpsk(s)

s = np.random.choice([0, 1], 32 * 2)
x = m.modulate_qpsk(s)
x_idft = ofdm.IDFT(x)

print("end")