
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


FILENAME = "df_week1_monday"
MODELS_DIR = "MODELS/"
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
DATA_PATH = MAIN_DIR + FILENAME
TIMESTEPS = 11
TIME_LENGTH = "MONDAY" # OR WEEK_1
FULL_NAME = f"{TIME_LENGTH}_T{TIMESTEPS}"

PROTO = "HTTP"
print("PROTO : ", PROTO)

tf.keras.backend.set_floatx('float64')


#############################################
# FUNCTIONS
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



data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_raw_{PROTO}.csv")
data = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")


look_back = TIMESTEPS
look_ahead = TIMESTEPS

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




# LOAD VAE MODEL



encoder = tf.keras.models.load_model(f"{MODELS_DIR}encoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5", 
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
decoder = tf.keras.models.load_model(f"{MODELS_DIR}decoder_vae_{TIME_LENGTH}_T{TIMESTEPS}_{PROTO}_FINAL.h5", 
                                  custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
vae = VAE(encoder, decoder, gamma=1)

z_mean, z_log_var = vae.encoder.predict(
    data[columns].values)
Z_sampling = Sampling()([z_mean, z_log_var]).numpy()




# LOAD GMM MODEL



# Load gmm model
gmm = joblib.load(f"{MODELS_DIR}gmm_{FULL_NAME}_{PROTO}.sav")

# Create sequence from the data
seq_labels = gmm.predict(Z_sampling)

# Define custom sequence
# seq_labels = [0, 1, 3, 1]



# SAMPLE SPECIFIC CLUSTER




# Apply sampling on GMM
def my_func(a, gmm):
    value = np.random.multivariate_normal(
        mean=gmm.means_[a[0]].reshape(-1), 
        cov=gmm.covariances_[a[0]], 
        size=None, check_valid='warn', tol=1e-8)
    return value


seq_labels_sample = np.apply_along_axis(
    my_func, 1, seq_labels.reshape(-1, 1), gmm)




# RE-CREATE FEATURES



vae_output_decoder = vae.decoder \
    .predict(seq_labels_sample)





