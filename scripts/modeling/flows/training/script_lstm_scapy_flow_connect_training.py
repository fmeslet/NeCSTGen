#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# import multiprocessing
import multiprocessing
import pickle

# Joblib
import joblib

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

# Personnal functions
# import functions

#############################################
# SET PARAMETERS
#############################################

FILENAME = "df_raw"
MODELS_DIR = "MODELS/"
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
DATA_PATH = MAIN_DIR + FILENAME
DATA_RANGE = [0, 1000000]
TIMESTEPS = 11

NB_CLUSTERS = 80 # -6
BLOC_LENGTH = 100
TIME_LENGTH = "MONDAY"
FULL_NAME = f"{TIME_LENGTH}_BL{BLOC_LENGTH}_T{TIMESTEPS}"

APP_LIST = ['ARP', 'LLC', 'LOOP', 'SNAP', 'TELNET', 
            'HTTP', 'SSH', 'SNMP', 'SMTP', 'DNS', 
            'NTP', 'FTP', 'RIP', 'IRC', 'POP', 'ICMP',
            'FINGER', 'TIME']

LIMIT_FLOW_PART = 4

PROTO = "TCP_GOOGLE_HOME"
print(f"PROTO : {PROTO}")

tf.keras.backend.set_floatx('float64')

#############################################
# USEFULL CLASS/FUNCTIONS
#############################################

def standardize(x, min_x, max_x, a, b):
  # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new

def unpacking_apply_along_axis(all_args):
    #(func1d, axis, arr, args, kwargs) = all_args
    func1d, axis, arr, args, kwargs = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

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
    def __init__(self, encoder, decoder,
                 gamma=0.5, **kwargs):
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
                keras.losses.mean_absolute_error(y_true=data_cont, 
                                                 y_pred=reconstruction), axis=(-1))
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
                    keras.losses.mean_absolute_error(y_true=data_cont, 
                                                     y_pred=reconstruction), axis=(-1))
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


class Predictor(keras.Model):
    def __init__(self, latent_dim,
                 feat_dim,
                 feat_dim_flags,
                 **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.feat_dim = feat_dim
        self.feat_dim_flags = feat_dim_flags

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_feat_tracker = keras.metrics.Mean(name="loss_feat")
        self.loss_flags_tracker = keras.metrics.Mean(name="loss_flags")

        self.decoder_inputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            self.latent_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        self.decoder_outputs = tf.keras.layers.Dense(self.feat_dim, activation="sigmoid")
        self.flags_outputs = tf.keras.layers.Dense(self.feat_dim_flags, activation="softmax")
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.latent_dim, #int(self.latent_dim), # Récupere l'output à la $
            activation='tanh', return_sequences=False, return_state=False))
        self.flatten = tf.keras.layers.Flatten()

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.loss_feat_tracker,
            self.loss_flags_tracker
        ]

    def call(self, inputs):
        data_output = inputs[1]

        #x = self.decoder_inputs(inputs[0])
        #x = self.lstm(x)

        x = self.lstm(inputs[0])
        x = self.flatten(x)
        x_feat = self.decoder_outputs(x)
        x_flags = self.flags_outputs(x)

        x = tf.concat(
            [x_flags, x_feat], axis=-1)

        target_flags = tf.slice(
            inputs[1], [0, 0], [-1, self.feat_dim_flags])
        target_feat = tf.slice(
            inputs[1], [0, self.feat_dim_flags], [-1, -1])

        loss_feat = tf.reduce_sum(
                tf.reduce_sum(
                    tf.keras.metrics.mean_squared_error(target_feat, x_feat), axis=(-1),
                )
            )
	
        loss_flags = tf.reduce_sum(
                    tf.reduce_sum(
                        keras.losses.categorical_crossentropy(target_flags, x_flags), axis=(-1),
                    )
                )

        loss = loss_flags + loss_feat

        self.loss_tracker.update_state(loss)
        self.loss_feat_tracker.update_state(loss_feat)
        self.loss_flags_tracker.update_state(loss_flags)

        return x
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data_input = data[0]
        data_output = data[1]

        with tf.GradientTape() as tape:

            x = self.lstm(data_input)
            x = self.flatten(x)

            x_feat = self.decoder_outputs(x)
            x_flags = self.flags_outputs(x)

            target_flags = tf.slice(
                data_output, [0, 0], [-1, self.feat_dim_flags])
            target_feat = tf.slice(
                data_output, [0, self.feat_dim_flags], [-1, -1])

            loss_feat = tf.reduce_sum(
                tf.reduce_sum(
                    tf.keras.metrics.mean_squared_error(target_feat, x_feat), axis=(-1),
                )
            )

            loss_flags = tf.reduce_sum(
                tf.reduce_sum(
                    keras.losses.categorical_crossentropy(target_flags, x_flags), axis=(-1),
                )
            )

            loss = loss_flags + loss_feat
   
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.loss_feat_tracker.update_state(loss_feat)
        self.loss_flags_tracker.update_state(loss_flags)

        return {
            "loss": self.loss_tracker.result(),
            "loss_feat": self.loss_feat_tracker.result(),
            "loss_flags": self.loss_flags_tracker.result()
        }



#############################################
# LAUNCH TRAINING
#############################################

# PREPARE DATA

if ("TCP_GOOGLE_HOME" in PROTO):
    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 
               'count_pkt', 'flags', 
               'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
               'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']
else:
    columns = ['layers_2', 'layers_3', 'layers_4', 'layers_5', 'flags', 'sport',
               'dport', 'length_total', 'time_diff', 'rate', "rolling_rate_byte_sec", 'rolling_rate_byte_min',
               'rolling_rate_packet_sec', 'rolling_rate_packet_min', 'header_length', 'payload_length']

data = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")
data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_raw_{PROTO}.csv")
arr = np.load(f"{MAIN_DIR}PROCESS/arr_process_{PROTO}_bit.npy", mmap_mode='r')

#arr = arr[:, 0:*8] # 605 normalement



# FLOW RE ORDERING



#condition_application = (data_raw['applications'] == PROTO)
#df_raw = data_raw[
#    condition_application].reset_index(drop=True).copy()
df_raw = data_raw.reset_index(drop=True)


# Extraction d'un dictonnaire avec flux -> taille
df_new = df_raw[['flow_id', 'timestamps']].groupby(
                ['flow_id']).min().rename(columns={'timestamps': 'flow_id_count'})

# Utilisation de map function pour flux -> taille
dict_map = df_new.to_dict()['flow_id_count']
df_raw['flow_count'] = df_raw['flow_id'].map(dict_map)

# Utilisation de map function pour timestamps + nombre de jour qui correpsond à l'ID
df_min = df_raw[['flow_id', 'timestamps']].groupby(
    ['flow_id']).min().rename(columns={'timestamps': 'min'})
df_min['pad'] = df_min.index.values*100000000
    #'min'].map(lambda x : np.random.randint(0, 5000, size=(1,))[0])
dict_pad = df_min.to_dict()['pad']

def some_func(a, b):
    #print(dict_pad[a])
    return b+dict_pad[a]

df_raw['timestamps_update'] = df_raw[['flow_id', 'timestamps']].apply(
    lambda x: some_func(a=x['flow_id'], b=x['timestamps']), axis=1)

# On sort en fonction de la timestamps (les flux sont autamiquement groupe)
df_raw = df_raw.sort_values(by=['timestamps_update'], ascending=True)
indexes_update = df_raw.index.values 

# J'applique l'insertion pour récuperer les index proprement
# Mettre les la numerotation des index
df_raw = df_raw.reset_index(drop=True)
#arr_update = arr[indexes_update]

## On extrait les index max de chaque flux (voir index result)
indexes = df_raw.duplicated(subset=['flow_id'], keep='first')
index_val = df_raw.index[~indexes].values
index_min = index_val.copy()

## On créer les index min de chaque flux (voir index result) ET on fait la diff
indexes = df_raw.duplicated(subset=['flow_id'], keep='last')
index_val = df_raw.index[~indexes].values
index_max = index_val.copy()

## On attribue les index_min et les index_max
df_new = df_raw[['flow_id']].drop_duplicates(subset=['flow_id'], keep='first')
df_new['index_min'] = index_min
df_new['index_max'] = index_max
df_new = df_new.set_index('flow_id')

dict_index_min = df_new[["index_min"]].to_dict()['index_min']
dict_index_max = df_new[["index_max"]].to_dict()['index_max']
df_raw['index_min'] = df_raw['flow_id'].map(dict_index_min)
df_raw['index_max'] = df_raw['flow_id'].map(dict_index_max)



# IDNETIFICATION FOR FIRST AND LAST PART



indexes_flow = df_raw[['index_min', 'index_max']] \
                .drop_duplicates(subset=['index_min', 'index_max'], 
                                 keep='first') \
                .reset_index(drop=True)
indexes_flow['quantity'] = indexes_flow['index_max'] - \
            indexes_flow['index_min']

# Extract the different index of flow
# The first and last four
for i in range(LIMIT_FLOW_PART):
    df_raw[f"index_min_{i}"] = df_raw['index_min']+i - df_raw.index.values
    indexes_flow[f"index_min_{i}"] = indexes_flow['index_min']+i
    
    df_raw[f"index_max_{i}"] = df_raw['index_max']-i - df_raw.index.values
    indexes_flow[f"index_max_{i}"] = indexes_flow['index_max']-i



# LOAD MODEL



encoder = tf.keras.models.load_model(f"{MODELS_DIR}encoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5",
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
decoder = tf.keras.models.load_model(f"{MODELS_DIR}decoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5", 
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
vae = VAE(encoder, decoder)



# PREDICT MODEL


# Reordering data to get same structure as df_concat
data = data.iloc[
     indexes_update.ravel()].reset_index(drop=True)
data_raw = data_raw.iloc[
     indexes_update.ravel()].reset_index(drop=True)


#z_mean, z_log_var = vae.encoder.predict(data.iloc[
#    indexes_update][columns].values)
z_mean, z_log_var = vae.encoder.predict(
         data[columns].values)
Z_sampling = Sampling()([z_mean, z_log_var]).numpy()



# CONCAT DATA



cols = [f"columns_{i}" for i in range(Z_sampling.shape[-1])]
df_sampling = pd.DataFrame() 

for i, col in enumerate(cols):
    df_sampling[col] = Z_sampling[:, i]
    #df_sampling[col] = z_mean[:, i]
df_concat = pd.concat([df_raw, df_sampling], axis=1)
# Pour obtenir le rank au sein du flow
df_concat['flow_index'] = df_concat.index.values - \
        df_concat['index_min']




# GENERATE SEQUENCE



# Limit index start and end
# Deux limites différentes complique le MILIEU de la
# génération du flux
min_val_gmm = 0
dict_gmm = {}

# Get flow values
cols = ['columns_0', 'columns_1']


# FIRST STEPS


# Generate start
for i in range(LIMIT_FLOW_PART):
    # Extract values
    cond = (df_concat[f'index_min_{i}'] == 0)
    arr_input = df_concat[cond][cols].values
    
    filename = f"{MODELS_DIR}gmm_{FULL_NAME}_{PROTO}_INDEX_MIN_{i}_FINAL.sav"
    gmm = joblib.load(filename)
    
    # Update for multiple clusters
    min_val_gmm_tmp = min_val_gmm
    for j in range(gmm.n_components):
        # Update dict
        dict_gmm[min_val_gmm_tmp] = [gmm, j]
        min_val_gmm_tmp += 1
    
    preds = gmm.predict(arr_input)
    preds += min_val_gmm 
    
    df_concat.loc[cond, 'cluster'] = preds
    
    # Update for the end
    min_val_gmm += gmm.n_components

    

# MIDDLE STEPS


# Generate Body
filename = f"{MODELS_DIR}gmm_{FULL_NAME}_{PROTO}_FINAL.sav"
gmm = joblib.load(filename)

# Update for multiple clusters
min_val_gmm_tmp = min_val_gmm
for j in range(gmm.n_components):
    # Update dict
    dict_gmm[min_val_gmm_tmp] = [gmm, j]
    min_val_gmm_tmp += 1

cond = True

for i in range(LIMIT_FLOW_PART):
    # Extract values
    cond = cond & ((df_concat[f'index_min_{i}'] != 0) &
            (df_concat[f'index_max_{i}'] != 0))
    
arr_input = df_concat[cond][cols].values

preds = gmm.predict(arr_input)
preds += min_val_gmm 

df_concat.loc[cond, 'cluster'] = preds

# Update for the end
min_val_gmm += gmm.n_components


# END STEPS

    
# Generate End
for i in range(LIMIT_FLOW_PART):
    # Extract values
    cond = (df_concat[f'index_max_{i}'] == 0)
    arr_input = df_concat[cond][cols].values
    
    filename = f"{MODELS_DIR}gmm_{FULL_NAME}_{PROTO}_INDEX_MAX_{i}_FINAL.sav"
    gmm = joblib.load(filename)
    
    # Update for multiple clusters
    min_val_gmm_tmp = min_val_gmm
    for j in range(gmm.n_components):
        # Update dict
        dict_gmm[min_val_gmm_tmp] = [gmm, j]
        min_val_gmm_tmp += 1
    
    preds = gmm.predict(arr_input)
    preds += min_val_gmm 
    
    df_concat.loc[cond, 'cluster'] = preds
    
    # Update for the end
    min_val_gmm += gmm.n_components



# SET BEGGINNING AND END OF A FLOW



# Define part of a flow 
df_concat["flow_part"] = 1

counter = 0
for i in range(LIMIT_FLOW_PART):
    cond = (df_concat[f'index_min_{i}'] == 0)
    df_concat.loc[cond, "flow_part"] = counter
    counter += (1/LIMIT_FLOW_PART)

counter = 0 
for i in range(LIMIT_FLOW_PART):
    cond = (df_concat[f'index_max_{i}'] == 0)
    df_concat.loc[cond, "flow_part"] = counter
    counter += (1/LIMIT_FLOW_PART)

print("[DEBUG] df_concat[flow_part] : ", df_concat["flow_part"])


# ECHANTILLONNAGE des clusters 



def my_func(a, dict_gmm):
    gmm = dict_gmm[a[0]][0]
    idx = dict_gmm[a[0]][1]
    value = np.random.multivariate_normal(
        mean=gmm.means_[idx].reshape(-1),
        cov=gmm.covariances_[idx],
        size=None, check_valid='warn', tol=1e-8)
    return value #pred_max

seq_labels = df_concat['cluster'].values
seq_labels_sample = np.apply_along_axis(
    my_func, 1, seq_labels.reshape(-1, 1), dict_gmm)



# ON DECODE AVEC VAE



vae_output_decoder = vae.decoder.predict(seq_labels_sample)
print("[DEBUG] vae.decoder.predict(seq_labels_sample) : ", vae_output_decoder.shape)


# EXTRACT REORDERING INDEXES



def my_func(i, indexes):
    index = np.where(
        indexes == i)[0]
    return index
    
indexes = np.apply_along_axis(
    my_func, 1, np.arange(
        0, indexes_update.size).reshape(-1, 1), 
    indexes_update)

#vae_output_decoder = vae_output_decoder #[indexes.ravel()]
#print("[DEBUG] indexes shape : ", indexes.shape)
#print("[DEBUG] vae_output_decoder[indexes] shape : ", vae_output_decoder.shape)

flow_part = df_concat['flow_part'].values.reshape(-1, 1) #.values[
          #indexes.ravel()].reshape(-1, 1)


# EXTRACT ADDITIONNALS FEATURES



# Extraire flags en version one hot
unique_values = data_raw['flags'].value_counts().index
values_tmp = data_raw['flags'].copy().values
values = data_raw['flags'].copy().values

for i, val in enumerate(unique_values):
    index = np.where(values == val)[0]
    values_tmp[index] = int(i)
flags_encoded = tf.keras.utils.to_categorical(values_tmp)


# Extraire la direction a partir des ports
# 1 si dport c'est 80
if (PROTO == "HTTP"):
    cond = (data_raw['sport'] == 80)
    data["direction"] = 0
    data.loc[cond, "direction"] = 1
elif (PROTO == "SNMP"):
    cond = (data_raw['sport'] == 161)
    data["direction"] = 0
    data.loc[cond, "direction"] = 1
elif (PROTO == "SMTP"):
    cond = ((data_raw['sport'] == 25) | (data_raw['sport'] == 587))
    data["direction"] = 0
    data.loc[cond, "direction"] = 1
else:
    df_flow_first = data_raw.groupby(
        "flow_id").first()

    # By default direction is 0
    data["direction"] = 0

    for i in range(df_flow_first.shape[0]):
        ip_src = df_flow_first.iloc[i]['ip_src']
        ip_dst = df_flow_first.iloc[i]['ip_dst']
        sport = df_flow_first.iloc[i]['sport']
        dport = df_flow_first.iloc[i]['dport']

        cond = ((data_raw['ip_src'] == ip_dst) &
                (data_raw['ip_dst'] == ip_src) &
                (data_raw['sport'] == dport) &
                (data_raw['dport'] == sport))

        data.loc[cond, "direction"] = 1




# RESET TIME DIFF




# For each flow id we set to zero (first index)
cond_first = (df_concat['index_min_0'] == 0)
df_concat.loc[cond_first, 'time_diff'] = 0

# Compute time diff for each flow and set Nan to 0
df_concat['time_diff'] = df_concat['timestamps'].diff(1)
df_concat.loc[cond_first, 'time_diff'] = 0


# Compute log
df_concat['time_diff_update'] = np.log10(
        df_concat['time_diff'].values)

#inf_val = np.log10(
#        data_raw['time_diff'].values).mean()
#inf_val = np.nan_to_num(
#        inf_val, neginf=0).mean()
#print("[DEBUG] inf_val : ", inf_val)

#df_concat.replace( # Imputation par la moyenne
#        -np.inf, inf_val, inplace=True)

df_concat['time_diff_update'] = np.nan_to_num(
        df_concat['time_diff_update'].values, neginf=0)

df_concat = df_concat.fillna(0)


# Normalize
df_concat['time_diff_update'] = standardize(
         df_concat['time_diff_update'], min_x=df_concat['time_diff_update'].min(),
         max_x=df_concat['time_diff_update'].max(), a=0, b=1)

# Set values to	data
data['time_diff'] = df_concat['time_diff_update'].values




# SELECT APPLICATIONS



look_back = TIMESTEPS
look_ahead = TIMESTEPS # A AUGMENTER !
range_fit = DATA_RANGE

X = data[columns].values #[range_fit[0]:range_fit[1]]
X_seq = create_windows(X, window_shape=look_back, end_id=-look_ahead)

X_idx = np.arange(0, X_seq.shape[0])
X_train_idx, X_val_idx, _, _ = sklearn.model_selection.train_test_split(X_idx, X_idx,
                                random_state=42, test_size=0.1,
                                shuffle=True)

print(f"X shape : {X.shape}")
print(f"X_seq shape : {X_seq.shape}")

X_train = X[X_train_idx].reshape(-1, X.shape[-1])
X_val = X[X_val_idx].reshape(-1, X.shape[-1])

print(f"X_train shape : {X_train.shape}")
print(f"X_val shape : {X_val.shape}")




# APPLY LSTM




look_back = TIMESTEPS
look_ahead = TIMESTEPS
range_fit_lstm = [0, X.shape[0]]

Z = vae_output_decoder
Z = np.concatenate((flow_part, vae_output_decoder), axis=-1)
#Z = Z[indexes_update] # <- UPDATE !!!

X = data[columns+['direction']].values 
X_seq = create_windows(X, window_shape=look_back, end_id=-look_ahead)

X_idx = np.arange(0, X_seq.shape[0])
X_train_idx, X_val_idx, _, _ = sklearn.model_selection.train_test_split(X_idx, X_idx,
                                random_state=42, test_size=0.1,
                                shuffle=True)


# Format y

y = data[['direction', 'length_total',
          'header_length',  'time_diff']].values
y = np.concatenate(
    (flags_encoded, y), axis=-1)
#y = y[indexes_update] # <- UPDATE !!!
y_seq = create_windows(
    y, window_shape=look_back, end_id=-look_ahead)


print("[DEBUG] Z shape : ", Z.shape)

Z_seq = create_windows(Z, window_shape=look_back, end_id=-look_ahead)

print(f"Z_seq shape : {Z_seq.shape}")
print(f"Z shape : {Z.shape}")

features_length = len(columns) + 1 # il faut ajouter flow_part !

Z_train = Z_seq[X_train_idx, :, :]
Z_val = Z_seq[X_val_idx, :, :]

Z_train[:, -1:, features_length:] = 0
Z_val[:, -1:, features_length:] = 0

y_train = y_seq[X_train_idx, -1]
y_val = y_seq[X_val_idx, -1]


print(f"Z_train shape : {Z_train.shape}")
print(f"Z_val shape : {Z_val.shape}")

print(f"y_train shape : {y_train.shape}")
print(f"y_val shape : {y_val.shape}")

gc.collect()

cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                         factor=0.5,
                         patience=1,
                         min_lr=1e-7,
                         min_delta=0.0001,
                         verbose=0,
                         skip_mismatch=True),
       tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                     min_delta=0.00001,
                     patience=1,
                     verbose=1,
                     mode='auto',
                     restore_best_weights=True)]



value_dim = int(Z_train.shape[-1])
lstm_vae_raw_feat = Predictor(latent_dim=value_dim, feat_dim=4, feat_dim_flags=flags_encoded.shape[-1])
# max_length+1 = car on sort aussi le time_diff ET Length total
lstm_vae_raw_feat.compile(
    optimizer=keras.optimizers.Adam(1e-3), run_eagerly=False)
history_lstm_vae_raw_feat = lstm_vae_raw_feat.fit([Z_train, y_train], 
                                                  validation_data=([Z_val, y_val], y_val),
                                                  epochs=50, 
                                                  batch_size=64, 
                                                  shuffle=True, 
                                                  use_multiprocessing=True,
                                                  callbacks=cbs)

print(history_lstm_vae_raw_feat.history)

#############################################
# SAVE MODEL and VALUES
#############################################

lstm_vae_raw_feat.save(f"{MODELS_DIR}LSTM_{FULL_NAME}_{PROTO}_{PROTO}_SCAPY_FLOW_CONNECT_FINAL", save_format='tf')
