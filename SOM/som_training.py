from minisom import MiniSom
import numpy as np


def train_network(som_shape, fit_data, sigma, learn_rate, epochs):
    best_inertia = None
    best_model = None
    best_clusters = None
    # train the som network as many time as indicated by the epoch
    for i in range(epochs):
        som = MiniSom(som_shape[0], som_shape[1], fit_data.shape[1],
                      sigma=sigma, learning_rate=learn_rate,
                      neighborhood_function='gaussian')
        # random initialisation of the weights
        som.random_weights_init(fit_data)
        som.train_random(fit_data, 10000)

        winner_coordinates = np.array([som.winner(x) for x in fit_data]).T
        clusters = np.ravel_multi_index(winner_coordinates, som_shape)

        # groups, centers = correspond_data(clusters, fit_data, som)
        # total_inertia = compute_inertia(groups, centers)
        total_inertia = som.quantization_error(fit_data)

        # save the model with the lowest quantization score
        if best_inertia is None or (total_inertia < best_inertia):
            best_inertia = total_inertia
            best_model = som
            best_clusters = clusters
    return best_inertia, best_model, best_clusters


# def correspond_data(clustering, trained_data, model):
#     keys = [i for i in range(len(np.unique(clustering)))]
#
#     data = [[trained_data[np.where(clustering == np.unique(clustering)[i])]
#              for i in range(len(np.unique(clustering)))]]
#
#     c = [np.array(centroid) for centroid in model.get_weights()]
#
#     data_groups = dict.fromkeys(keys, data)
#     centroids = dict.fromkeys(keys, c)
#     return data_groups, centroids
#
#
# def compute_inertia(data_groups, centroids):
#     sum_up = []
#     for key in centroids:
#         diff = np.abs(np.subtract(data_groups[key][0][0], centroids[key][0]))
#         sum_up.append(np.sum(diff, axis=1))
#     inertia = np.sum(sum_up)
#     return inertia
