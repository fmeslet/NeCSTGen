#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# Avoid bug
from sklearn.cluster import KMeans, DBSCAN
from itertools import groupby
from sklearn.neighbors import KernelDensity
from scapy.utils import PcapWriter
from scapy.layers import *
from collections import *

import os

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
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

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
DATA_RANGE = [0, 1000000]
TIMESTEPS = 11

NB_CLUSTERS = 80 # -6
BLOC_LENGTH = 100
TIME_LENGTH = "MONDAY"
FULL_NAME = f"{TIME_LENGTH}_BL{BLOC_LENGTH}_T{TIMESTEPS}"

RESULTS_DIR = "RESULTS/"

START_INDEX = 0
END_INDEX = 20_000

FORWARD_STEPS = 7

PROTO = "UDP_GOOGLE_HOME"
print(f"PROTO : {PROTO}")

# Define version or details about file
EXT_NAME = "_FINAL"
print(f"EXT_NAME : {EXT_NAME}")

tf.keras.backend.set_floatx('float64')

#############################################
# USEFULL CLASS/FUNCTIONS
#############################################

def transform_packet_int_bytes(packet_int):
    packet_bytes = [(int(packet_int[i])).to_bytes(1, byteorder='big') for i in range(packet_int.shape[0])]
    #packet_bytes_array = np.array(packet_int)

    #packet_int_array_pad = np.reshape(packet_int_array_pad, (1536, 1))
    return packet_bytes

def transform_packet_bit_int(packet_bit):
    packet_int = []
    for i in range(8, len(packet_bit)+1, 8):
        packet_bit_str = packet_bit[i-8:i].astype(str)
        packet_bit_str = "".join(packet_bit_str)
        packet_int.append(int(packet_bit_str, 2))
    return packet_int

class Generator():
    def __init__(self, vae, 
                 limit_predict):
        self.vae = vae
        self.dict_algo = {}
        self.gmm = {}
        self.limit_predict = limit_predict
        self.seq_labels = None
        self.seq_labels_ravel = None
        
    ########################
    # PLOT CLUSTER FUNCTION
    ########################
    
    def plot_corr_latent(self, df):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        z_mean, z_log_var = self.vae.encoder.predict(
            df.values)
        data_latent = Sampling()([z_mean, z_log_var]).numpy()

        df_latent = pd.DataFrame(data_latent, columns=["latent_1", "latent_2"]) 
        df_all = pd.concat([df, df_latent], axis=1)

        sns.heatmap(np.around(df_all.corr(), decimals=2), annot=True)
    
    def plot_scatter(self, X, s=.8):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.scatter(x=X[:, 0], y=X[:, 1], s=s)
        ax.legend()
        #sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax, hue=labels)
        ax.set_title("Scatter plot")
    
    def plot_kmeans(self, X, s=.8):
        labels = self.predict_kmeans(X)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(x=X[:, 0], y=X[:, 1], 
                    c=labels, s=s)
        ax.legend()
        #sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax, hue=labels)
        ax.set_title("Scatter plot of KMeans")
        
    def plot_gmm(self, X, index, title, s=.8):
        means = self.dict_algo['gmm'].means_
        covariances = self.dict_algo['gmm'].covariances_
        Y_ = self.dict_algo['gmm'].predict(X)
        
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
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
    
    ####################
    # GENERATOR FUNCTION
    ####################
    
    def get_data_cluster(self):
        data_cluster = pd.DataFrame()
        data_cluster['x'] = generator.dict_algo['gmm'].means_[:, 0]
        data_cluster['y'] = generator.dict_algo['gmm'].means_[:, 1]
        data_cluster['labels'] = np.arange(0,  generator.dict_algo['gmm'].means_.shape[0])
        return data_cluster
    
    def get_sample(self, classe):
        # 10 for security
        sample_quantity = 5000 #int((1/self.dict_algo['gmm'].weights_[classe])*quantity + 
        X, y = self.dict_algo['gmm'].sample(sample_quantity)
        indexes = np.where(y == classe)[0]
        return X[indexes[0]]
    
    def set_joint_count(self, labels, 
                        #labels_all_list,
                        mat_count, forward_steps=3):
        dict_proba = defaultdict(list)

        # Add a values at the end and remove it
        labels = np.append(labels, 0)
        labels_windows = create_windows(
            labels.reshape(-1, 1), window_shape=min(
                labels.size-1, forward_steps), end_id=-1)
        #labels = labels[0:-1]

        def my_func(a, unique_labels):
            idx = []
            for el in a:
                idx.append(np.where(
                    el == self.unique_labels)[0])
                
            idx = np.array(idx).ravel()
            return idx

        index_labels = np.apply_along_axis(
            my_func, 0, labels_windows, self.unique_labels)

        # Prepare array for slicing
        index_slicing = []
        for step in range(index_labels.shape[1]):
            index_slicing.append(index_labels[:, step, 0])
        index_slicing = np.array(index_slicing)

        def my_func(a, mat_count):
            idx = a.reshape(
                a.shape[0], 1).tolist()
            mat_count[idx] = mat_count[idx] + 1
        
        np.apply_along_axis(
            my_func, 1, [index_slicing], mat_count)
        
        
    def fit_generator(self, seq_labels, forward_steps=3):
        
        self.seq_labels = []
        
        labels_all = []
        
        for i in range(len(seq_labels)):
            self.seq_labels.append(
                seq_labels[i])
            labels_all.extend(seq_labels[i])
        
        self.unique_labels, _ = np.unique(
                labels_all, return_counts=True)
        
        #forward_steps = min(forward_steps, seq_labels.shape[-1])
        self.joint_count = np.zeros(
            (len(self.unique_labels),)*forward_steps)
        
        self.set_joint_count(labels=labels_all, #self.seq_labels[i], 
                             mat_count=self.joint_count, 
                             forward_steps=forward_steps)
            
    def get_proba(self, idx, i, 
                  length):
        
        count = self.joint_count[idx.tolist()]
        count = np.squeeze(count)

        # On somme les axes jusqu'a obtenir une ligne...
        if(len(count.shape) > 1):
            count = count.sum(
                axis=tuple(i for i in range(len(count.shape)))[1:]) 
        
        proba = count / count.sum(axis=-1)
        
        return proba
        
        
    def predict_generator(self, length=20, init_start=True):
        timesteps = len(self.joint_count.shape)-1
        
        # /!\ AUCUNE garantie que la matrice de transition
        # mènera à tous les états
        
        dict_index_label = {}
        dict_label_index = {}
        for i, l in enumerate(self.unique_labels):
            dict_index_label[i] = l
            dict_label_index[l] = i
        
        sample_label = []
        sample_index = [] # get index of each element in array
        
        # Generate first step
        i = 0
        while(len(sample_label) != length):
            
            # FIRST STEP
            if (i == 0):
                if (init_start):
                    # Look the first cluster of all seq_labels
                    labels = [self.seq_labels[j][
                        0] for j in range(len(self.seq_labels))]
                    labels_choice = np.random.choice(
                        labels, size=1)
                    
                    # On stock le premier index (on force l'init)
                    sample_label.append(
                        labels_choice.ravel()[0])
                    sample_index.append(
                        dict_label_index[sample_label[-1]])
                    
                    idx = np.array(sample_index[-timesteps:])
                    idx = idx.reshape(idx.size, -1) # Reshape for index format
                else:
                    # Init with all
                    idx = np.arange(
                        0, self.joint_count.shape[0])
            else:
                if (timesteps == 0):
                    # Init with all
                    idx = np.arange(
                        0, self.joint_count.shape[0])
                else:
                    idx = np.array(sample_index[-timesteps:])
                    idx = idx.reshape(idx.size, -1) # Reshape for index format
                
            i = len(sample_label)
            proba = self.get_proba(
                idx, i, length)
            
            index_label = np.array(
                list(dict_label_index.values()), 
                dtype=np.uint32)
            index = np.random.choice(
                    index_label, 1, p=proba)[0]
            
            sample_index.append(index)
            sample_label.append(dict_index_label[sample_index[-1]])

            i = len(sample_label)

        return sample_label



def transform_packet_bytes_int(packet_bytes, length=1522):
    packet_int = [int(byte) for byte in packet_bytes]
    packet_int_array = np.array(packet_int)

    packet_int_array_pad = np.lib.pad(packet_int_array,
                            (0,length-packet_int_array.shape[0]),
                            'constant', constant_values=(0))

    packet_int_array_pad = np.reshape(packet_int_array_pad, (length, 1))
    return packet_int_array_pad

def standardize(x, min_x, max_x, a, b):
  # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new

def transform_packet_bit_bytes(packet_bit):
    packet_int = np.apply_along_axis(transform_packet_bit_int, 1, packet_bit)
    packet_bytes = transform_packet_int_bytes(packet_int[0])
    return packet_bytes

def standardize(x, min_x, max_x, a, b):
  # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new

def gen_seq_labels(generator,
                   df_raw,
                   start_idx,
                   end_idx,
                   forward_steps=2):
    """Fit the transition matrix.

    Args:
        generator (Generator): _description_
        df_raw (pd.DataFrame): _description_
        start_idx (int): _description_
        end_idx (int): _description_
        forward_steps (int, optional): Number of steps used to condition the transition matrix. Defaults to 2.

    Returns:
        _type_: _description_
    """
    
    seq_true = []
        
    num_cluster = df_raw.iloc[
        start_idx:end_idx]['cluster'].values

    # Add to clusters list
    seq_true.append(num_cluster)

    # Generate series
    generator.fit_generator(
        seq_labels=seq_true, 
        forward_steps=forward_steps)

    return generator



def gen_pcap(model,
             generator,
             df_raw,
             start_idx,
             end_idx,
             forward_steps=2,
             nb_packet_gen=10,
             filename="test_1_bis.pcap"):

    # Get PCAP generation
    timesteps = 11 # 11

    if (nb_packet_gen is None):
        update_nb_packet_gen = True
    else:
        update_nb_packet_gen = False

    # Get label sequence
    # *2 pour bloc_length_gen pour eviter les problème de paquet bizarre au début...
    generator = gen_seq_labels(generator=generator,
                               df_raw=df_raw,
                               start_idx=start_idx,
                               end_idx=end_idx,
                               forward_steps=forward_steps)

    
    df_ts = pd.DataFrame()
    seq_pred = []

    if (update_nb_packet_gen):
        nb_packet_gen = end_idx - start_idx

    #print("[DEBUG][gen_pcap] flow_length_gen : ", flow_length_gen)

    # Generate series
    seq_pred = generator.predict_generator(
        length=nb_packet_gen)

    print("[DEBUG][gen_pcap] nb_packet_gen : ", nb_packet_gen)

    # Apply GMM
    def my_func(a, gmm):
        value = np.random.multivariate_normal(
            mean=gmm.means_[a[0]].reshape(-1),
            cov=gmm.covariances_[a[0]],
            size=None, check_valid='warn', tol=1e-8)
        return value #pred_max


    # GENERATE FEAT
    seq_pred = np.array(seq_pred)
    print("[DEBUG][gen_pcap] seq_pred.shape : ", seq_pred.shape)
    df_feat = np.apply_along_axis(
            my_func, 1, seq_pred.reshape(-1, 1),
            gmm)

    #print("[DEBUG][gen_pcap] df_feat : ", df_feat)

    #print("[DEBUG][gen_pcap] df_feat shape : ", df_feat.shape)

    num_packets = df_feat.shape[0]

    #print("[DEBUG][gen_pcap] num_packets : ", num_packets)

    # Prepare input data
    inputs_tmp = df_feat.copy()
    inputs_tmp = np.concatenate(
        (np.zeros((timesteps-1, df_feat.shape[-1])), inputs_tmp), axis=0) # timesteps-1
    inputs_tmp = np.concatenate(
        (inputs_tmp, np.zeros((1, df_feat.shape[-1]))), axis=0)

    inputs = create_windows(
        inputs_tmp, window_shape=timesteps, end_id=-1)

    #print("[DEBUG][gen_pcap] inputs shape : ", inputs.shape)

    df_ts_tmp = write_to_pcap(model=model,
                          inputs=inputs, 
                          df_raw=df_raw[columns],
                          df_feat=df_feat, 
                          num_packets=num_packets,
                          filename=filename)
    #df_ts_tmp['flow_id'] = i
    df_ts_tmp = df_ts_tmp.iloc[
        timesteps:].reset_index(drop=True)

    df_ts = pd.concat(
        [df_ts, df_ts_tmp], axis=0).reset_index(drop=True)

    return seq_pred, df_ts


def write_to_pcap(model, inputs, df_feat, df_raw,
                  num_packets, filename="test.pcap"):
    
    
    timesteps = inputs.shape[1]
    num_df_feat = df_feat.shape[-1]
    
    inputs = inputs.copy()
    
    count_all_vae = []
    timesteps_all_vae = []
    time_diff_all_vae = []
    
    # Creat padding for first random flows
    # It's not important if timesteps are 0 at the beginning
    timesteps_all_vae.extend([0]*timesteps)
    time_diff_all_vae.extend([0]*timesteps)
    count_all_vae.extend([0]*timesteps)
    
    # + timesteps pour empecher l'ecriture de chose "bizarre"
    for i in range(1, num_packets): #range(timesteps, num_packets):

        # Get data
        inputs = inputs[0:1, -timesteps:, :]
        
        gc.collect()
        
        #columns = ['flow_id_count', 'length_total_sum',
        #           'time_diff', 'rate', 'length_total_std',
        #           'header_length_std', 'payload_length_std']
        
        if ((PROTO == "UDP_GOOGLE_HOME") or (PROTO == "TCP_GOOGLE_HOME")):
            time_raw_vae = df_feat[i-1][0]
            count_raw_vae = df_feat[i-1][1]
        else:
            time_raw_vae = df_feat[i-1][1]
            count_raw_vae = df_feat[i-1][0]        


        # SAVE TIME DIFF
        time_diff_vae = standardize(
                       time_raw_vae, min_x=0, #df_feat[i, -2], min_x=0, #+timesteps au lieu de 8
                       max_x=1, a=df_raw['time_diff'].min(), # [condition_ports] ?
                       b=df_raw['time_diff'].max())
        time_diff_vae = np.array([10**time_diff_vae])
        time_diff_vae[time_diff_vae == -np.inf] = 0
        time_vae = time_diff_vae + timesteps_all_vae[-1]
        time_vae = time_vae[0]
        timesteps_all_vae.append(time_vae)
        time_diff_all_vae.append(time_diff_vae[0])
        
        
        # SAVE PACKET COUNT
        count_vae = standardize(
                       count_raw_vae, min_x=0, # +timesteps
                       max_x=1, a=df_raw['flow_id_count'].min(),
                       b=df_raw['flow_id_count'].max())
        count_vae = np.array([10**count_vae])
        count_vae[count_vae == -np.inf] = 0
        count_vae = count_vae[0]
        count_all_vae.append(count_vae)
        
        
        #print("[DEBUG][write_to_pcap] df_feat[i] shape : ", df_feat[i].shape)
        
        # Prepare next input
        next_input = np.reshape(df_feat[i], (1, 1, num_df_feat))
        inputs = np.concatenate((inputs, next_input), axis=1)
        
        #print("[DEBUG][write_to_pcap] next_input shape : ", next_input.shape)
        #print("[DEBUG][write_to_pcap] inputs shape : ", inputs.shape)
    
    data_return = pd.DataFrame({"count_total" : count_all_vae,
                                "time_diff" : time_diff_all_vae,
                                 "timesteps" : timesteps_all_vae})
    return data_return


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


class Processing():
    def __init__(self, df, arr, columns_enc=['layers_0', 'layers_1', 
                                       'layers_2', 'layers_3', 
                                       'layers_4', 'layers_5', 
                                       'flags', 'sport',
                                       'dport', 'applications'], 
                 columns_add=['length_total', 'time_diff', 
                              'rate', 'rolling_rate_byte_sec', 'rolling_rate_byte_min',
                               'rolling_rate_packet_sec', 'rolling_rate_packet_min']):
        self.df_raw = df
        self.arr_raw = arr
        
        self.df_process = self.df_raw[
            columns_enc+columns_add].copy()
        self.X = None
        
        self.columns_enc =  columns_enc
        self.columns_add =  columns_add
        
        self.dict_le = {}
        
    def transform_le(self, col, normalize):
        le = preprocessing.LabelEncoder()
        self.dict_le[col] = le
        self.df_process[col] = le.fit_transform(self.df_process[col])
        
        if (normalize and len(le.classes_)>1):
            self.df_process[col] = standardize(
              self.df_process[col], min_x=self.df_process[col].min(),
                max_x=self.df_process[col].max(), a=0, b=1)
            
        return le
              
    
    def process(self, normalize=True):
        
        for col in self.columns_add:
            self.df_process[col] = standardize(
                  self.df_process[col], min_x=self.df_process[col].min(),
                   max_x=self.df_process[col].max(), a=0, b=1)
          
        if (('sport' in self.columns_enc) and ('dport' in self.columns_enc)):
            for col in ['sport', 'dport']:
                condition_sport = (self.df_process[col] < 1024)
                self.df_process.loc[~condition_sport, col] = 0
            
        for col in self.columns_enc:
            self.transform_le(
                col=col, normalize=True)

    def get_data_learning(self): 
        return self.df_process[self.columns]
    
    def transform_array_to_df(self, x): 
        # Digitize need to be done manually
        # X_new = self.digitize(x)
        
        # Transforme to df
        df = pd.DataFrame()
        for i, col in enumerate(self.columns_enc+self.columns_add):
            df[col] = np.reshape(
                x, (-1, len((self.columns_enc+self.columns_add))))[:, i]
        
        # Reverse standardization
        df = self.reverse_standardize(df)
        
        # Reverse label encoding
        df = self.reverse_transform_le(df)
                
        return df
    
    def reverse_transform_le(self, df):
        df_new = df.copy()
        for col in self.columns_enc:
            df_new[col] = self.dict_le[col].inverse_transform(
                df_new[col].values.ravel().astype(int))
        return df_new
                
    def digitize(self, x):
        unique = self.df_process[self.columns_enc].nunique().values
        x_digitize = np.empty((x.shape[0], x.shape[1], 0))
        for i in range(len(self.columns_enc)):
            bins = np.linspace(0, 1, unique[i])
            centers = (bins[1:]+bins[:-1])/2

            feat_digitize = bins[np.digitize(x[:, :, i:i+1], centers)]
            x_digitize = np.concatenate((x_digitize, feat_digitize), axis=-1)
        
        # Voit si on standardize les deux dernière colonnes ?
        x_digitize = np.concatenate((x_digitize, x[:, :, -2:]), axis=-1)
        return x_digitize
    
    def reverse_standardize(self, df):
        # Creer un tableau pour récolter les résultats
        df_new = pd.DataFrame()
        df_copy = df.copy()
        
        # For encoded features
        for col in self.columns_enc:    
            df_new[col] = standardize(
                       df_copy[col], min_x=0,
                       max_x=1, a=0,
                       b=len(self.dict_le[col].classes_)-1)

        # For columns added
        for col in self.columns_add:    
            df_new[col] = standardize(
                       df_copy[col], min_x=0,
                       max_x=1, a=self.df_raw[col].min(),
                       b=self.df_raw[col].max())

        return df_new


##################
# LOAD DATA
#################


data_raw = pd.read_csv(f"DATA/PROCESS/df_raw_flows_{PROTO}.csv")
data = pd.read_csv(f"DATA/PROCESS/df_process_flows_{PROTO}.csv")

columns = ['flow_id_count', 'length_total_sum',
   'time_diff', 'rate', 'length_total_std',
   'header_length_std', 'payload_length_std']


#############
# PROCESSING
#############


df_raw = data_raw.copy() #[condition_ports].reset_index(drop=True)

for c in  ['flow_id_count', 'length_total_sum',
            'time_diff', 'rate']:

    df_raw[c] = np.log10(
        data_raw[c].values)

    inf_val = df_raw[c].iloc[1:].mean()
    
    df_raw.replace( # Mean imputation
        -np.inf, inf_val, inplace=True)
    
df_raw = df_raw.fillna(0)

processing = Processing(df=df_raw, 
                        arr=None,
                        columns_enc=[],
                        columns_add=columns)

processing.process(normalize=True)

data = processing.df_process.fillna(0)


###################
# LOAD GMM
###################

# Note pretty small error to correct...
if((PROTO == "UDP_GOOGLE_HOME") or (PROTO == "TCP_GOOGLE_HOME")):
    X = data[['time_diff', 
              'flow_id_count']].values
else:
    X = data[['flow_id_count',
              'time_diff']].values

gmm = joblib.load(f"{MODELS_DIR}gmm_{FULL_NAME}_{PROTO}_FLOWS_FINAL.sav")
seq_labels = gmm.predict(X) # Generate clusters sequence

df_concat = pd.DataFrame()
df_concat['cluster'] = seq_labels

df_raw = pd.concat([df_raw, df_concat], axis=1)
data = pd.concat([data, df_concat], axis=1)

generator = Generator(vae=None, limit_predict=4)
generator.gmm = gmm


#############################################
# GENERATION
#############################################


nb_packet_gen = 10000 # e.q to packet when flow are not present

result = gen_pcap(model=None,
                  df_raw=df_raw.copy(),
                  generator=generator,
                  start_idx=START_INDEX,
                  end_idx=END_INDEX,
                  nb_packet_gen=nb_packet_gen, 
                  forward_steps=FORWARD_STEPS)

result[1].to_csv(f"{RESULTS_DIR}DF_GEN_FLOW_{PROTO}{EXT_NAME}.csv", index=False)

print(f"[DEBUG] filename : {RESULTS_DIR}DF_GEN_FLOW_{PROTO}{EXT_NAME}.csv")
print("[DEBUG] result[0] : ", result[0])
print("[DEBUG] result[1] : ", result[1])

print("[DEBUG] START_INDEX : ", START_INDEX)
print("[DEBUG] END_INDEX : ", END_INDEX)
