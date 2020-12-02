from simulations.snr_ber_average_ibo import Result #pklを使うのに必要
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pickle
import logging

# シミュレーション結果の保存先を作成する
dt_now = datetime.datetime.now()
dirname = '../results/snr_ber_average_ibo_chain_load/' + dt_now.strftime("%Y/%m/%d/%H_%M_%S")
# dirname = '../results/keep/momentam_batch_nn_15_full_2/merge/'
os.makedirs(dirname, exist_ok=True)

formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(filename=dirname + '/log.log', level=logging.INFO, format=formatter)
logging.info('start')
logging.info(dt_now.strftime("%Y/%m/%d %H:%M:%S"))

# グラフ
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 22
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

results = []
pkl_paths = [
    '../results/snr_ber_average_ibo/2020/11/18/19_30_23/snr_ber_average_ibo.pkl',
    '../results/snr_ber_average_ibo/2020/11/18/19_30_27/snr_ber_average_ibo.pkl',
    '../results/snr_ber_average_ibo/2020/11/18/19_39_16/snr_ber_average_ibo.pkl',
    '../results/snr_ber_average_ibo/2020/11/18/19_39_17/snr_ber_average_ibo.pkl',
    # '../results/snr_ber_average_ibo/2020/11/18/15_54_23/snr_ber_average_ibo.pkl',
    # '../results/snr_ber_average_ibo/2020/11/18/15_54_28/snr_ber_average_ibo.pkl',
    # '../results/snr_ber_average_ibo/2020/11/18/16_07_44/snr_ber_average_ibo.pkl',
    # '../results/snr_ber_average_ibo/2020/11/18/16_07_46/snr_ber_average_ibo.pkl',
]

for pkl_path in pkl_paths:
    with open(pkl_path, 'rb') as f:
        logging.info("loaded_pkl: %s" % pkl_path)
        result = pickle.load(f)
        results.append(result)

# 結合させる
errors_list = []
losss_list = []
val_losss_list = []

for result in results:
    errors_list.append(result.errors)
    losss_list.append(result.losss)
    val_losss_list.append(result.val_losss)

errors = np.concatenate(errors_list, 2)
losss = np.concatenate(losss_list, 2)
val_losss = np.concatenate(val_losss_list, 2)

logging.info('chainload')
logging.info(len(results))

params = results[0].params
logging.info('params')
logging.info(params)

snrs_db = np.linspace(params['SNR_MIN'], params['SNR_MAX'], params['SNR_NUM'])

# %%
# グラフ作成
# SNR-BERグラフ
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("BER")
ax.set_yscale('log')
ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
y_min = pow(10, 0)
y_max = pow(10, -6)
ax.set_ylim(y_max, y_min)
ax.set_xlim(params['SNR_MIN'], params['SNR_MAX'])
ax.grid(linestyle='--')

train_data = params['n'] - (params['n'] * params['trainingRatio'])
n_ave = train_data * params['SNR_AVERAGE'] * len(results)

color_list = ["r", "g", "b", "c", "m", "y", "k", "w"]
for IBO_index, IBO_db in enumerate(params['IBO_dB']):
    errors_sum = np.sum(errors[IBO_index], axis=1)
    bers = errors_sum / n_ave
    ax.plot(snrs_db, bers, color=color_list[IBO_index], marker='o', linestyle='--', label="IBO=%d[dB]" % IBO_db)

ax.legend()
plt.savefig(dirname + '/SNR_BER.pdf')
plt.show()

# Plot learning curve
for index, snr_db in enumerate(snrs_db):
    plt.figure()

    for IBO_index, IBO_db in enumerate(params['IBO_dB']):
        loss_avg = np.mean(losss[IBO_index][index], axis=0).T
        val_loss_avg = np.mean(val_losss[IBO_index][index], axis=0).T
        plt.plot(np.arange(1, len(loss_avg) + 1), loss_avg, color=color_list[IBO_index], marker='o', linestyle='--',
                 label='Training Frame (IBO=%d[dB])' % IBO_db)
        plt.plot(np.arange(1, len(loss_avg) + 1), val_loss_avg, color=color_list[IBO_index + len(params['IBO_dB'])],
                 marker='o', linestyle='--', label='Test Frame (IBO=%d[dB])' % IBO_db)

    plt.ylabel('less')
    plt.yscale('log')
    plt.xlabel('Training Epoch')
    plt.legend()
    plt.grid(which='major', alpha=0.25)
    plt.xlim([0, params['nEpochs'] + 1])
    plt.xticks(range(1, params['nEpochs'], 2))
    plt.savefig(dirname + '/snr_db_' + str(snr_db) + '_NNconv.pdf', bbox_inches='tight')

## 13, 16試行がおかしい
# nn_model = results[0].nn_models[0][5][13]
# system_model = nn_model.system_model
# print('h_si:', system_model.h_si)
# print('h_s:', system_model.h_s)
#
# nn_model = results[0].nn_models[0][5][16]
# system_model = nn_model.system_model
# print('h_si:', system_model.h_si)
# print('h_s:', system_model.h_s)
#
# system_models = []
# for result in results:
#     for trials_index in range(params['SNR_AVERAGE']):
#         system_models.append(result.nn_models[0][5][trials_index].system_model)
#
# print(system_models)