from src.previous_research.nn import NNModel as PreviousNNModel
import src.previous_research.fulldeplex as fd
import matplotlib.pyplot as plt
from simulations.common import graph
import numpy as np


if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research_load'

    params = {
        'samplingFreqMHz': 20,  # Sampling frequency, required for correct scaling of PSD
        'hSILen': 13,  # Self-interference channel length
        'trainingRatio': 0.9,  # Ratio of total samples to use for training
        'dataOffset': 14,  # Data offset to take transmitter-receiver misalignment into account
        'nHidden': 17,  # Number of hidden layers in NN
        'nEpochs': 20,  # Number of training epochs for NN training
        'learningRate': 0.004,  # Learning rate for NN training
        'batchSize': 32,  # Batch size for NN training
    }
    x, y, noise, measuredNoisePower = fd.loadData('../src/previous_research/data/fdTestbedData20MHz10dBm', params)

    previous_nn_model = PreviousNNModel(
        params['hSILen'],
        params['nHidden'],
        params['learningRate']
    )

    previous_nn_model.learn(
        x,
        y,
        params['trainingRatio'],
        params['hSILen'],
        params['nEpochs'],
        params['batchSize']
    )

    loss = previous_nn_model.history.history['loss']
    val_loss = previous_nn_model.history.history['val_loss']

    graph.init_graph()
    learn_fig, learn_ax = graph.new_learning_curve_canvas(params['nEpochs'])

    epoch = np.arange(1, len(loss) + 1)
    learn_ax.plot(epoch, loss, color="k", marker='o',
                  linestyle='--',
                  label='Training Frame')
    learn_ax.plot(epoch, val_loss,
                  color="r",
                  marker='o', linestyle='--', label='Test Frame')

    learn_ax.legend()
    plt.savefig('../results/keep/previous_research_load_data/less.pdf', bbox_inches='tight')