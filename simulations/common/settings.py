import os
import datetime
import logging
import matplotlib.pyplot as plt


def init_simulation_output(dirname: str = ''):
    # シミュレーション結果の保存先を作成する
    os.makedirs(dirname, exist_ok=True)

    init_log(dirname + '/log.log')
    init_graph()


def dirname_current_datetime(identifier: str) -> str:
    dt_now = datetime.datetime.now()
    return '../results/' + identifier + '/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")


def init_log(filename: str):
    formatter = '%(levelname)s : %(asctime)s : %(message)s'
    logging.basicConfig(filename=filename, level=logging.INFO, format=formatter)
    logging.info('start')


def init_graph():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 22
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plt_color_list() -> list:
    return ["r", "g", "b", "c", "m", "y", "k", "w"]
