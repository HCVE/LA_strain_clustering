from hyperopt import Trials, STATUS_OK, hp, fmin, tpe, rand
import som_training as st
import numpy as np


# tune the network's hyper-parameters based on the quantization error
def quantization_error_tuning(space):
    _, model, _ = st.train_network(som_shape=(space['x'], space['y']), fit_data=space['fit_data'], sigma=space['sig'],
                                   learn_rate=space['learning_rate'], epochs=20)
    val = model.quantization_error(space['fit_data'])
    return {'loss': val, 'status': STATUS_OK}


# tune the network's hyper-parameters based on the distance of the weights.
# Not used for the purposes of the paper "Unsupervised Time Series Analysis of
# Left Atrium Strain for Cardiovascular Risk Assessment"
def weight_distance_tuning(space):
    _, model, _ = st.train_network(som_shape=(space['x'], space['y']), fit_data=space['fit_data'], sigma=space['sig'],
                                   learn_rate=space['learning_rate'], epochs=20)

    number_of_clusters = len(np.unique([model.winner(x) for x in space['fit_data']]).T)
    val = 1 / sum(model.distance_map()) + (space['x'] - number_of_clusters)
    return {'loss': val, 'status': STATUS_OK}


def begin_fine_tune(som_shape, interp_strain):
    # define the searching space of the hyper-parameters
    spaces = {'sig': hp.uniform('sig', 0.1, min(som_shape)),
              'learning_rate': hp.uniform('learning_rate', 0.1, 5),
              'x': som_shape[0], 'y': som_shape[1], 'fit_data': interp_strain}

    trials = Trials()
    # retrieve the best model my minimising the quantization error
    best = fmin(fn=quantization_error_tuning, space=spaces, algo=tpe.suggest, max_evals=100,
                trials=trials)
    return trials, best


# alternative function to fine tune the hyper-parameters. It is not used for the purposes of the paper
# "Unsupervised Time Series Analysis of Left Atrium Strain for Cardiovascular Risk Assessment"
def tune_hyperparameters(som_shape, fit_data, sigma, learn_rate):
    best_inertia = None
    model = None
    best_sigma = None
    best_rate = None
    best_clusters = None
    # test for several
    for i in range(len(sigma)):
        for j in range(len(learn_rate)):
            inertia, som_model, clusters = st.train_network(som_shape=som_shape, fit_data=fit_data,
                                                            sigma=sigma[i], learn_rate=learn_rate[j], epochs=1)
            if best_inertia is None or (inertia < best_inertia):
                best_inertia = inertia
                best_sigma = sigma[i]
                best_rate = learn_rate[j]
                best_clusters = clusters
                model = som_model
    return best_inertia, best_sigma, best_rate, best_clusters, model
