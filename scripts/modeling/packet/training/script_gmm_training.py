#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# Avoid bugs
from itertools import groupby
from collections import *

# Garbage collector
import gc

# Linear algebra and data processing
import numpy as np
import pandas as pd
import math
import random

# Get version
from platform import python_version
import sklearn
import tensorflow as tf

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


#############################################
# PARAMETERS
#############################################


FILENAME = "df_week1_monday"
MODELS_DIR = "MODELS/"
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
DATA_PATH = MAIN_DIR + FILENAME
TIMESTEPS = 11
TIME_LENGTH = "MONDAY" # OR WEEK_1
FULL_NAME = f"{TIME_LENGTH}_T{TIMESTEPS}"

PROTO = "HTTP" # DEfine the protocol used
print("PROTO : ", PROTO)

# Set Tensorflow to float64 (by default set to float32)
tf.keras.backend.set_floatx('float64')


#############################################
# FUNCTIONS
#############################################


def create_windows(data, window_shape, step = 1, start_id = None, end_id = None):
    """Apply sliding window on the data and reshape it.

    Args:
        data (np.array): the data.
        window_shape (int): size of the window applied on data.
        step (int, optional): apply the sliding. Defaults to 1.
        start_id (int, optional): first inex inside the data to start the sliding windos. Defaults to None.
        end_id (_type_, optional): end index inside the data to stop the sliding window. Defaults to None.

    Returns:
        np.array: the data sliced format to a matrix.
    """
    
    data = np.asarray(data)
    data = data.reshape(-1,1) if np.prod(data.shape) == max(data.shape) else data
        
    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id
    
    data = data[int(start_id):int(end_id),:]
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))
    
    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)
    
    return np.squeeze(window_data, 1)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.

    Args:
        layers (tf.keras.layers.Layer): layers class.
    """
    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
class VAE(keras.Model):
    """Class which define a Beta Variational Autoencoder.

    Args:
        keras (tensorflow.keras.Model): Keras model.
    """
    def __init__(self, encoder, decoder, gamma=0.5, **kwargs):
        """Constructor.

        Args:
            encoder (tensorflow.keras.Model): Encoder part of the Variational Autoencoder.
            decoder (tensorflow.keras.Model): Decoder part of the Variational Autoencoder.
            gamma (float, optional): Beta paramter of a Beta-VAE (Beta Variational Autoencoder). 
            Defaults to 0.5 (for a classical Variational Autoencoder).
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.gamma = gamma
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.sampling = Sampling()

    @property
    def metrics(self):
        """Return the metrics used.

        Returns:
            list: Array of the metrics used.
        """
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        """Send the data at input and give back the output of the model.

        Args:
            inputs (numpy.array): Data give as input of the model.

        Returns:
            numpy.array: Data give as output of the model.
            Reconstruction of the input data. 
        """
        data_input = inputs

        # Get he mean and variance
        z_mean, z_log_var = self.encoder(data_input)
        z = self.sampling([z_mean, z_log_var])

        reconstruction_raw = self.decoder(z)
        
        reconstruction = tf.reshape(reconstruction_raw, [-1, 1])
        data_cont = tf.reshape(data_input, [-1, 1])

        reconstruction_loss_0 = tf.reduce_sum(
                keras.losses.binary_crossentropy(y_true=data_cont, y_pred=reconstruction), axis=(-1))
        reconstruction_loss_1 = tf.reduce_sum(
                keras.losses.mean_absolute_error(y_true=data_cont, y_pred=reconstruction), axis=(-1))
        reconstruction_loss = reconstruction_loss_0 + reconstruction_loss_1
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.gamma * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        loss = reconstruction_loss + kl_loss

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return reconstruction_raw

    def train_step(self, data):
        """Perform the backpropagation during the training and send back the performance obtained.

        Args:
            data (numpy.array): Data give as input of the model for training.

        Returns:
            dict: The name of the metrics used (keys) and the values obtained (values).
        """
        if isinstance(data, tuple):
            data = data[0]

        data_cont = data[0]
        data_shift = data[1]
        
        with tf.GradientTape() as tape:

            z_mean, z_log_var = self.encoder(data_cont)
            z = self.sampling([z_mean, z_log_var])

            reconstruction = self.decoder(z)

            reconstruction = tf.reshape(reconstruction, [-1, 1])
            data_cont = tf.reshape(data_cont, [-1, 1])

            reconstruction_loss_0 = tf.reduce_sum(
                    keras.losses.binary_crossentropy(y_true=data_cont, y_pred=reconstruction), axis=(-1))
            reconstruction_loss_1 = tf.reduce_sum(
                    keras.losses.mean_absolute_error(y_true=data_cont, y_pred=reconstruction), axis=(-1))
            reconstruction_loss = reconstruction_loss_0 + reconstruction_loss_1

            # Loss for first part
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.gamma * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "reconstruction_loss_cont_0": reconstruction_loss_0,
            "reconstruction_loss_cont_1": reconstruction_loss_1,
        }
    
def build_encoder_dense(nb_feat, input_shape):
    """Create an encoder for a Variational Autoencoder (VAE).

    Args:
        nb_feat (int): Dimension of the latent space.
        input_shape (tuple): Shape of the input layer.

    Returns:
        tensorflow.Keras.Model: The encoder model. 
    """
    latent_dim = nb_feat
    encoder_inputs_0 = keras.Input(shape=(input_shape,))
    x = encoder_inputs_0

    x = layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(4, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs_0, [z_mean, z_log_var], name="encoder")
    
    return encoder

def build_decoder_dense(nb_feat, input_shape):
    """Create a decoder for a Variational AutoEncoder (VAE).

    Args:
        nb_feat (int): Dimension of the latent space. 
        input_shape (tuple): Shape of the input layers.

    Returns:
        tensorflow.Keras.Model: The decoder model.
    """

    latent_dim = nb_feat
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(4, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(latent_inputs)
    x = layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    decoder_outputs = layers.Dense(input_shape, activation="sigmoid")(x)

    decoder = keras.Model(latent_inputs, outputs=decoder_outputs, name="decoder")
    
    return decoder


def plot_gmm(X, title, model, s=.8):
    """Plot data with the covariances matrices associate 
    to each clusters define the Gaussian Mixture Model (GMM). 

    Args:
        X (numpy.array): Data to plot.
        title (string): Title of the plot.
        model (sklearn.mixture): Clustering model (Gaussian Mixture Model).
        s (float, optional): Size of the point inside the scatter plot. Defaults to .8.
    """
    means = model.means_
    covariances = model.covariances_
    Y_ = model.predict(X)
    
    color_iter = itertools.cycle(['navy', 'c', 
                                  'cornflowerblue', 
                                  'gold',
                                  'darkorange'])
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=s, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle)#, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    ax.set_title(title)



#############################################
# CODE
#############################################


# PREPARE DATA


# If data come from darpa dataset
if (PROTO in ["HTTP", "SMTP", "DNS", "SNMP"]):

    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 'flags', 'sport',
               'dport', 'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
                'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

# If data come from Google Home dataset
elif(PROTO == "TCP_GOOGLE_HOME"):

    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 'count_pkt',
           'flags', 'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
            'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

elif(PROTO == "TCP_GOOGLE_HOME"):

    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', # 'count_pkt',
            'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
            'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

# If data come from LoRaWAN dataset
elif ((PROTO == "LORA_10") or 
      (PROTO == "LORA_1")):

    columns = ['fport', 'mtype', 'code_rate', 'size', 'bandwidth',
               'spreading_factor', 'frequency', 'crc_status', 
               'length_total', 'time_diff', 'snr', 'rssi', 
                'rate', 'rolling_rate_byte_sec', 'rolling_rate_byte_min',
                'rolling_rate_packet_sec', 'rolling_rate_packet_min',
                'header_length', 'payload_length', 'fcnt']


# Load the data
data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_raw_{PROTO}.csv")
data = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")

look_back = TIMESTEPS
look_ahead = TIMESTEPS

# Extract usefull columns inside the data
X = data[columns].values

# Split the data. We use the create windows function, even if it's not necessary,
# to keep consistency with 
X_seq = create_windows(X, window_shape=look_back, end_id=-look_ahead)
X_idx = np.arange(0, X_seq.shape[0])
X_train_idx, X_val_idx, _, _ = sklearn.model_selection.train_test_split(X_idx, X_idx,
                                random_state=42, test_size=0.1,
                                shuffle=True) # , stratify=y


print(f"X shape : {X.shape}")
print(f"X_seq shape : {X_seq.shape}")

X_train = X[X_train_idx].reshape(-1, X.shape[-1])
X_val = X[X_val_idx].reshape(-1, X.shape[-1])

print(f"X_train shape : {X_train.shape}")
print(f"X_val shape : {X_val.shape}")


# LOAD VAE MODEL


encoder = tf.keras.models.load_model(f"{MODELS_DIR}encoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5", 
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
decoder = tf.keras.models.load_model(f"{MODELS_DIR}decoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5", 
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
vae = VAE(encoder, decoder, gamma=1)

# Project the data into a latent space wit the Variational Auencoder.
z_mean, z_log_var = vae.encoder.predict(
    data[columns].values)
Z_sampling = Sampling()([z_mean, z_log_var]).numpy()


# VISUALIZE THE LATENT SPACE


# Visualize the various cluster
# fig, ax = plt.subplots(1, 1, figsize=(7, 7))
# ax.scatter(x=Z_sampling[:, 0], y=Z_sampling[:, 1], s=0.1)


# TRAIN GMM


# Apply mean init if cluster
# are note well identified
#means_init = np.array([[-1.5, 0],
#                       [-1.45, -1],                       
#                       [-1.45, 0.4],
#                       [-0.7, 1]])


gmm = GaussianMixture(n_components=25,
                      #n_components=means_init.shape[0],
                      covariance_type='full',
                      init_params='kmeans',
                      max_iter=700,
                      #means_init=means_init
                     )

# Fit model
gmm.fit(Z_sampling)

# For plotting cluster sequence
seq_labels = gmm.predict(Z_sampling)


# PLOT GAUSSIAN (with covariance matrices)
# set up on each cluster
plot_gmm(Z_sampling,
         'Gaussian Mixture', 
         model=gmm)

# Train again the GMM if the clusters are note
# well defined 


