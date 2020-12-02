import os
import datetime
import logging
import matplotlib.pyplot as plt
import argparse
import json


def init_simulation(simulation_name: str) -> (dict, str):
    args = parse_args()
    output_dir = get_output_dir(args, simulation_name)
    params = get_params(args, simulation_name)

    init_output(output_dir)

    return params, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs')
    parser.add_argument('-o', '--output_dir')
    return parser.parse_args()


def get_output_dir(args: argparse.Namespace, simulation_name: str) -> str:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = dirname_current_datetime(simulation_name)

    return output_dir


def get_params(args: argparse.Namespace, simulation_name: str) -> dict:
    # パラメータ
    configs = args.configs
    if configs is None:
        with open('configs/%s.json' % simulation_name) as f:
            params = json.load(f)
    else:
        params = json.loads(configs)
    return params


def init_output(dirname: str = ''):
    # シミュレーション結果の保存先を作成する
    os.makedirs(dirname, exist_ok=True)

    init_log(dirname + '/log.log')
    init_graph()


def dirname_current_datetime(identifier: str) -> str:
    dt_now = datetime.datetime.now()
    # workdir = os.environ['PYTHONPATH']
    workdir = '..'
    return '%s/results/%s/%s' % (workdir, identifier, dt_now.strftime("%Y/%m/%d/%H_%M_%S"))


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
