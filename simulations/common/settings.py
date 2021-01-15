import os
import datetime
import logging
import argparse
import json
from simulations.common import graph
from simulations.common import slack
from src.nn import NNModel
import numpy as np
import time


def init_simulation(simulation_name: str, ofdm: bool = False) -> (dict, str):
    args = parse_args()
    output_dir = get_output_dir(args, simulation_name)
    params = get_params(args, simulation_name)

    init_output(output_dir)
    logging.info('start')
    logging.info(output_dir)
    logging.info(params)

    # 再現性を出す為にseedを記録しておく
    set_seed(params)
    if ofdm is False:
        # 学習ビット数とビット数とテストビット数を記録する。
        set_simulation_bits_to_params(params)

    dump_params(params, output_dir)

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
        param_path = 'configs/%s.json' % simulation_name
        params = load_param(param_path)
    else:
        params = json.loads(configs)
    return params


def load_param(param_path: str) -> dict:
    with open(param_path) as f:
        return json.load(f)


def dump_params(params: dict, output_dir: str):
    with open(output_dir + '/params.json', 'w') as f:
        json.dump(params, f, indent=4)


def init_output(dirname: str = ''):
    # シミュレーション結果の保存先を作成する
    os.makedirs(dirname, exist_ok=True)

    init_log(dirname + '/log.log')
    graph.init_graph()


def dirname_current_datetime(identifier: str) -> str:
    dt_now = datetime.datetime.now()
    # workdir = os.environ['PYTHONPATH']
    workdir = '..'
    return '%s/results/%s/%s' % (workdir, identifier, dt_now.strftime("%Y/%m/%d/%H_%M_%S"))


def init_log(filename: str):
    formatter = '%(levelname)s : %(asctime)s : %(message)s'
    logging.basicConfig(filename=filename, level=logging.INFO, format=formatter)


def finish_simulation(params: dict, output_dir: str, output_png_path: str = None):
    if output_png_path is not None:
        slack.upload_file(output_png_path, "end: " + output_dir + "\n" + json.dumps(params, indent=4))
    logging.info("end")


def set_seed(params: dict):
    seed = params.get('seed')
    if seed is None:
        seed = int(time.time())

    params["seed"] = seed
    np.random.seed(seed)


def set_simulation_bits_to_params(params: dict):
    params["train_bits"] = NNModel.train_bits(params["n"], params['trainingRatio'])
    params["test_bits"] = NNModel.test_bits(params["n"], params['trainingRatio'], params['h_si_len'])
