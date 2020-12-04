from src.previous_research.nn import NNModel as PreviousNNModel

if __name__ == '__main__':
    SIMULATIONS_NAME = 'previous_research'

    params = {
        "chan_len": 13,
        "n_hidden": 17,
        "learning_rate": 0.004,
        "batch_size": 32
    }
    previous_nn_model = PreviousNNModel(13, 17, 0.004)
