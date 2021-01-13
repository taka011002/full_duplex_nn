import numpy as np


def modulate_qpsk(data: np.ndarray) -> np.ndarray:
    """
    QPSK変調を行う．

    :param data: [1, 0, 1, 0, ,,,]
    :return:
    """
    data = np.where(data > 0, 1, -1)  # BPSK変調
    data = data / np.sqrt(2)
    i = data[0::2]  # 奇数番目は実部
    q = data[1::2]  # 偶数番目は虚部
    return i + 1j * q  # 平均電力が1になるようにルート2をかける


def demodulate_qpsk(signal: np.ndarray) -> np.ndarray:
    """
    QPSK復調を行う．
    :param signal:
    :return:
    """
    odd_bit = np.where(signal.real > 0, 1, 0)
    even_bit = np.where(signal.imag > 0, 1, 0)

    data = np.zeros(2 * signal.size, dtype=int)
    data[0::2] = odd_bit
    data[1::2] = even_bit

    return data


def iq_imbalance(x: np.ndarray, gamma: float = 0.0, phi: float = 0.0, selective: bool = False) -> np.ndarray:
    """
    IQIを行う．現状は周波数非選択性のみ

    :param x:
    :param gamma:
    :param phi:
    :param selective:
    :return:
    """
    if not selective:
        alpha = np.cos(phi) + 1j * gamma * np.sin(phi)
        beta = gamma * np.cos(phi) + 1j * np.sin(phi)
        return alpha * x + beta * np.conj(x)

    # TODO 周波数選択性IQIを実装する


def sspa_rapp_ibo(input_signal: np.ndarray, IBO_dB: int = 0, rho: float = 0.5, ofdm: bool = False) -> np.ndarray:
    """
    入力バックオフ(IBO)によって飽和電力を定めたSSPA(Rappモデル)の値を取得する．

    :param input_signal:
    :param IBO_dB:
    :param rho:
    :return:
    """
    ibo = 10 ** (IBO_dB / 10)  # IBOをもとにアンプの飽和電力を決める
    size = input_signal.shape[0]
    if ofdm == True:
        shape = input_signal.shape
        size = shape[0] * shape[1] # サブキャリア数をブロック数全ての要素で割る
    P_in = np.sum((input_signal * input_signal.conj()).real) / size  # nで割るべき？
    A = np.sqrt(P_in * ibo)
    return sspa_rapp(input_signal, A, rho)


def a_sat(input_signal: np.ndarray, IBO_dB: int = 0) -> float:
    ibo = 10 ** (IBO_dB / 10)  # IBOをもとにアンプの飽和電力を決める
    P_in = np.sum((input_signal * input_signal.conj()).real) / input_signal.shape[0]  # nで割るべき？
    A = np.sqrt(P_in * ibo)
    return A


def sspa_rapp(input_signal: np.ndarray, saturation: float = 1, rho: float = 0.5) -> np.ndarray:
    """
    RappモデルによるSSPAの値を求める．

    :param input_signal:
    :param saturation:
    :param rho:
    :return:
    """
    AM = np.abs(input_signal)
    AMtoAM = AM / np.power(1 + np.power(AM / saturation, 2 * rho), 1 / (2 * rho))
    phase = np.angle(input_signal)
    amp_output = AMtoAM * np.exp(1j * phase)
    return amp_output


def channel(size: int = 1, length: int = 1, scale: float = 1.0) -> np.ndarray:
    """
    複素ガウス分布に従った周波数非選択性通信路を生成する．
    生成する通信路の要素を全て違う値にする際はsizeを指定してあげる．
    指定しない場合は，1つの通信路のみ生成する．
    周波数選択性通信路の場合は，電力を合わせる為に，チャネル長さlengthを渡してあげる．

    :param x:
    :param length:
    :return:
    """
    variance = np.reciprocal(np.sqrt(2 * length))
    h = np.random.normal(loc=0, scale=scale, size=(size, length)) + 1j * np.random.normal(loc=0, scale=scale,
                                                                                          size=(size, length))
    h = h * variance
    return h


def exponential_decay_channel(size: int = 1, length: int = 1, scale: float = 1.0, alpha: float = 1.0,  p_0: float = 1.0) -> np.ndarray:
    """
    指数減衰モデルの周波数選択性通信路を生成する．
    生成する通信路の要素を全て違う値にする際はsizeを指定してあげる．
    指定しない場合は，1つの通信路のみ生成する．
    """
    l = length - 1
    l_array = np.arange(l + 1)
    p_lambda = p_0 / np.sum(np.exp(-1 * alpha * l_array))
    P = p_lambda * np.exp(-1 * alpha * l_array)

    h = np.random.normal(loc=0, scale=scale, size=(size, length)) + 1j * np.random.normal(loc=0, scale=scale, size=(size, length))
    variance = np.sqrt(2)
    h = np.sqrt(P) * h / variance
    return h


def awgn(size: tuple, sigma: float) -> np.ndarray:
    """
    AWGNの値を取得する

    :param size:
    :param sigma:
    :return:
    """
    variance = np.reciprocal(np.sqrt(2))
    v = np.random.normal(loc=0, scale=sigma, size=size) + 1j * np.random.normal(loc=0, scale=sigma, size=size)
    v = v * variance
    return v


def snr_db(snr_min: int = 0, snr_max: int = 25, snr_num: int = 6):
    return np.linspace(snr_min, snr_max, snr_num)


def to_exact_number(db):
    """
    デシベルから真値へ変化する。

    :param db: int
    :return: int
    """
    return 10 ** (db / 10)


def sigmas(snrs: np.ndarray) -> np.ndarray:
    """
    SNR(db)を元sigmaを求める。

    :param snrs: ndarray
    :return: ndarray
    """
    inv_two_sigma_squares = to_exact_number(snrs)
    sigma_squares = np.reciprocal(inv_two_sigma_squares)
    return np.sqrt(sigma_squares)


def check_error(origin: np.ndarray, target: np.ndarray) -> float:
    """
    誤り率を判定する．

    :param origin:
    :param target:
    :return:
    """
    error = np.sum(origin != target)
    ber = error / origin.size

    return ber
