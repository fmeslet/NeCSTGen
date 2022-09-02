
#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

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
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Personnal functions
# import functions

#############################################
# SET PARAMETERS
#############################################

FILENAME = "df_week1_monday"
MODELS_DIR = "MODELS/"
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
DATA_PATH = MAIN_DIR + FILENAME
DATA_RANGE = [0, 200000]
TIMESTEPS = 11
TIME_LENGTH = "MONDAY" # OR WEEK_1
FULL_NAME = f"{TIME_LENGTH}_T{TIMESTEPS}"

PROTO = "HTTP"
print("PROTO : ", PROTO)

tf.keras.backend.set_floatx('float64')

#############################################
# USEFULL CLASS/FUNCTIONS
#############################################

def train_val_test_split(X, y, random_state, train_size=0, 
                         val_size=0, test_size=0):
    
    # Prendre le cas de la stratification
    # Prendre en cmpte la spération...
    X = np.arange(0, X.shape[0])
    y = np.arange(0, y.shape[0])
    train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(X, y,
                                random_state=random_state, test_size=1-train_size, 
                                shuffle=True) # , stratify=y

    # Get data test from val
    X = X[val_idx]
    y = y[val_idx]
    val_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(X, y,
                                random_state=random_state, test_size=0.5, 
                                shuffle=True)
    
    return train_idx, val_idx, test_idx

def create_windows(data, window_shape, step = 1, start_id = None, end_id = None):
    
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
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    
class VAE(keras.Model):
    def __init__(self, encoder, decoder, gamma=0.5, **kwargs):
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
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def call(self, inputs):
        # ONLY FOR LSTM
        data_input = inputs#[0]
        #data_shift = inputs[1]

        z_mean, z_log_var = self.encoder(data_input)
        z = self.sampling([z_mean, z_log_var])
        # ONLY FOR LSTM
        reconstruction_raw = self.decoder(z)
        #reconstruction_raw, states = self.decoder([z, data_shift])
        
        reconstruction = tf.reshape(reconstruction_raw, [-1, 1]) # Avant 1 : 200
        data_cont = tf.reshape(data_input, [-1, 1])

        reconstruction_loss_0 = tf.reduce_sum(
                keras.losses.binary_crossentropy(y_true=data_cont, y_pred=reconstruction), axis=(-1))
        reconstruction_loss_1 = tf.reduce_sum(
                keras.losses.mean_absolute_error(y_true=data_cont, y_pred=reconstruction), axis=(-1))
        reconstruction_loss = reconstruction_loss_0 + reconstruction_loss_1
        #reconstruction_loss = reconstruction_loss_1
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = self.gamma * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        loss = reconstruction_loss + kl_loss

        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        #return reconstruction_raw, states
        return reconstruction_raw

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        #data_input = data[0]
        data_cont = data[0]
        data_shift = data[1]
        #data_output = data[2]
        
        with tf.GradientTape() as tape:

            #print(tf.shape(data_cont))
            #print(tf.shape(data_cat))

            z_mean, z_log_var = self.encoder(data_cont)
            z = self.sampling([z_mean, z_log_var])
            # ONLY FOR LSTM
            #reconstruction, states = self.decoder([z, data_shift])
            reconstruction = self.decoder(z)
            #reconstruction = reconstruction
            
            #print(tf.shape(reconstruction))

            reconstruction = tf.reshape(reconstruction, [-1, 1]) # Avant 1 : 200
            data_cont = tf.reshape(data_cont, [-1, 1])

            reconstruction_loss_0 = tf.reduce_sum(
                    keras.losses.binary_crossentropy(y_true=data_cont, y_pred=reconstruction), axis=(-1))
            reconstruction_loss_1 = tf.reduce_sum(
                    keras.losses.mean_absolute_error(y_true=data_cont, y_pred=reconstruction), axis=(-1))
            reconstruction_loss = reconstruction_loss_0 + reconstruction_loss_1
            #reconstruction_loss = reconstruction_loss_1

            # Loss for first part
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.gamma * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss# + kl_loss_output

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
    latent_dim = nb_feat#50*nb_feat
    encoder_inputs_0 = keras.Input(shape=(input_shape,))
    #x = layers.Flatten()(encoder_inputs_0)
    x = encoder_inputs_0

    x = layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(4, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs_0, [z_mean, z_log_var], name="encoder")
    
    return encoder

def build_decoder_dense(nb_feat, input_shape):
    latent_dim = nb_feat#50*nb_feat
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(4, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(latent_inputs)
    x = layers.Dense(6, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    decoder_outputs = layers.Dense(input_shape, activation="sigmoid")(x)
    #decoder_outputs = tf.reshape(x, [-1, 100, nb_feat])

    decoder = keras.Model(latent_inputs, outputs=decoder_outputs, name="decoder")
    
    return decoder

#############################################
# LAUNCH TRAINING
#############################################



# PREPARE DATA



# If data come from darpa dataset
if (PROTO in ["HTTP", "SMTP", "DNS", "SNMP"]):

    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 'flags', 'sport',
               'dport', 'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
                'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

# If data come from Google Home dataset
elif(PROTO == "TCP_GOOGLE_HOME"):

    print("JE PASSE PAR TCP")
    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 'count_pkt',
           'flags', 'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
            'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

elif(PROTO == "TCP_GOOGLE_HOME"):

    print("JE PASSE PAR UDP")
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



data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_raw_{PROTO}.csv")
data = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")


look_back = TIMESTEPS
look_ahead = TIMESTEPS # A AUGMENTER !
range_fit = DATA_RANGE

X = data[columns].values

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


# CREATE MODEL AND TRAIN


gc.collect()
nb_feat = 2#data.columns.shape[0]
encoder = build_encoder_dense(
    nb_feat=nb_feat, input_shape=X.shape[-1])
decoder = build_decoder_dense(
    nb_feat=nb_feat, input_shape=X.shape[-1])

cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                         factor=0.5,
                         patience=1,
                         min_lr=1e-7,
                         min_delta=0.1,
                         verbose=0,
                         skip_mismatch=True)]#,
       #tf.keras.callbacks.EarlyStopping(monitor='val_loss',
       #              min_delta=0.00005,
       #              patience=1,
       #              verbose=1,
       #              mode='auto',
       #              restore_best_weights=True)]

vae = VAE(encoder, decoder, gamma=1)
vae.compile(optimizer=keras.optimizers.Adam(1e-4), run_eagerly=False) # False
history_vae = vae.fit([X_train, X_train], 
                      validation_data=(X_val, X_val),
                      epochs=50, # 100
                      batch_size=32, 
                      shuffle=True,
                      use_multiprocessing=True,
                      callbacks=cbs)

print(history_vae.history)

#############################################
# SAVE MODEL and VALUES
#############################################

encoder.save(f"{MODELS_DIR}encoder_vae_{FULL_NAME}_{PROTO}_FINAL.h5")
decoder.save(f"{MODELS_DIR}decoder_vae_{FULL_NAME}_{PROTO}_FINAL.h5")
