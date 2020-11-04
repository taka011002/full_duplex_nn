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


def sspa_rapp_ibo(input_signal: np.ndarray, IBO_dB: int = 0, rho: float = 0.5) -> np.ndarray:
    """
    入力バックオフ(IBO)によって飽和電力を定めたSSPA(Rappモデル)の値を取得する．

    :param input_signal:
    :param IBO_dB:
    :param rho:
    :return:
    """
    ibo = 10 ** (IBO_dB / 10)  # IBOをもとにアンプの飽和電力を決める
    P_in = np.sum(np.abs(input_signal)) / input_signal.size
    A = np.sqrt(P_in * ibo)
    return sspa_rapp(input_signal, A, rho)


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


def channel(size: int = 1, length: int = 0) -> np.ndarray:
    """
    周波数非選択性通信路を生成する．

    :param x:
    :param length:
    :return:
    """
    variance = np.reciprocal(np.sqrt(2 * (length + 1)))
    # scale = np.sqrt(variance)
    # size = x.size

    # TODO 周波数選択性の場合は複数hをベクトルで生成する
    h = np.random.normal(loc=0, scale=1, size=size) + 1j * np.random.normal(loc=0, scale=1, size=size)
    h = h * variance
    return h


def awgn(size: int, sigma: float) -> np.ndarray:
    """
    AWGNの値を取得する

    :param size:
    :param sigma:
    :return:
    """
    v = np.random.normal(loc=0, scale=sigma, size=size) + 1j * np.random.normal(loc=0, scale=sigma, size=size)
    v = v / np.sqrt(2)
    return v


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
    error = np.sum(origin != target)
    ber = error / origin.size

    return ber