import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

"""
Build the Deep CNN Autoencoder Network using optuna library to find the optimal network parameters
This module requires as input the trial parameters from optuna, the data for the training and 
a path to save the model produced in each trial
"""


def custom_loss(y_true, y_pred):
    # custom loss function to calculate the reconstruction error of the decoder's output
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=-1)
    return mse


def CNN_autoencoder(trial, input_data, latent_dim):
    conv_layers = trial.suggest_categorical('Convolutional Layer', [1, 2, 3, 4, 5])
    hidden_layers = trial.suggest_categorical('Hidden Layer', [2, 3, 4, 5])
    conv_filters = []
    conv_kernels = []
    activation = []
    neurons = []

    # encoder
    encoder_inputs = keras.Input(shape=(input_data.shape[1], 1))
    filters = trial.suggest_categorical('f1', [16, 32, 64])
    kernels = trial.suggest_categorical('k1', [3, 5, 7])
    activation_function = trial.suggest_categorical('activations', ['relu', 'elu', 'linear'])

    conv_filters.append(filters)
    conv_kernels.append(kernels)
    activation.append(activation_function)

    encoder = layers.Conv1D(filters, kernels, activation=activation_function, padding='same')(encoder_inputs)
    for i in range(conv_layers - 1):
        filters = trial.suggest_categorical('f' + str(i + 2), [16, 32, 64])
        kernels = trial.suggest_categorical('k' + str(i + 2), [3, 5, 7])
        encoder = layers.Conv1D(filters, kernels, activation=activation_function, padding='same')(encoder)
        conv_filters.append(filters)
        conv_kernels.append(kernels)

    encoder = layers.Flatten()(encoder)
    for i in range(hidden_layers):
        number_of_neurons = trial.suggest_categorical('l' + str(i + 1), [16, 32, 64])
        encoder = layers.Dense(number_of_neurons, activation=activation_function)(encoder)
        neurons.append(number_of_neurons)

    z = layers.Dense(latent_dim, name="z_mean")(encoder)

    # decoder
    decoder = layers.Dense(neurons[-1], activation=activation_function)(z)
    for i in range(hidden_layers-2):
        decoder = layers.Dense(neurons[-2-i], activation=activation_function)(decoder)

    decoder = layers.Dense(input_data.shape[1] * conv_filters[-1], activation=activation_function)(decoder)
    decoder = layers.Reshape((input_data.shape[1], conv_filters[-1]))(decoder)

    for i in range(conv_layers):
        decoder = layers.Conv1DTranspose(conv_filters[-1-i], conv_kernels[-1-i], activation=activation_function,
                                         padding='same')(decoder)
    decoder_output = layers.Conv1DTranspose(1, conv_kernels[0], activation=activation_function, padding='same')(decoder)
    return keras.Model(encoder_inputs, decoder_output, name='AutoEncoder')


def objective(trial, fit_data, path):
    """
    :param trial: optuna trial with a combination of CNN parameters
    :param fit_data: strain curves on which the CNN will be trained on
    :param path: path to save the models for each optuna trial
    :return:
    """
    latent_dim = trial.suggest_categorical('latent dimension', [2, 3, 4, 5])
    autoencoder = CNN_autoencoder(trial, fit_data, latent_dim)

    autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, patience=10)
    autoencoder.fit(fit_data, fit_data, epochs=30, batch_size=32, validation_split=0.15, shuffle=True, verbose=0,
                    callbacks=[es])
    autoencoder.save(path+"Trial Models/{}.h5".format(trial.number))
    return es.best
