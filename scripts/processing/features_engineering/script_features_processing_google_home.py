
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""
DATA_DIR_REF = "/users/rezia/fmesletm/DATA_GENERATION/DATA/DATA_RAW/GOOGLE_HOME/media_pcap_anonymized/"
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/"

PROTO = "TCP" # or UDP


#######################
# FUNCTIONS / CLASS
#######################


# From : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
def standardize(x, min_x, max_x, a, b):
    """Standardize data between the range of [a, b].

    Args:
        x (np.array): Data to standardize.
        min_x (int): Minimum value for standardize.
        max_x (int): Maximum value for standardize.
        a (int): Lower limit of the range.
        b (int): Upper limit of the range.

    Returns:
        np.array: Data standardize.
    """
    # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new


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
        
        for col in tqdm(self.columns_add):
            self.df_process[col] = standardize(
                  self.df_process[col], min_x=self.df_process[col].min(),
                   max_x=self.df_process[col].max(), a=0, b=1)
          
        if (('sport' in self.columns_enc) and ('dport' in self.columns_enc)):
            for col in ['sport', 'dport']:
                condition_sport = (self.df_process[col] < 1024)
                self.df_process.loc[~condition_sport, col] = 0
            
        for col in tqdm(self.columns_enc):
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


#######################
# CODE
#######################



df_raw = pd.read_csv("./DATA/PROCESS/df_raw_{PROTO}_GOOGLE_HOME.csv")


for c in ['time_diff', 'rate', 'rolling_rate_byte_sec', 
              'rolling_rate_byte_min', 'rolling_rate_packet_sec', 
              'rolling_rate_packet_min']:

    df_raw[c] = np.log10(
        df_raw[c].values)

    inf_val = df_raw[c].iloc[1:].mean()
    
    df_raw.replace( # Mean imputations
        -np.inf, inf_val, inplace=True)

    
columns_enc = ['layers_0', 'layers_1', 
               'layers_2', 'layers_3', 
               'layers_4', 'layers_5', 
               'flags', "count_pkt",
               #'sport','dport', 
               'applications']
df_raw[columns_enc] = df_raw[columns_enc].astype(str)
columns_add=['length_total', 'time_diff', 
              'rate', 'rolling_rate_byte_sec', 'rolling_rate_byte_min',
              'rolling_rate_packet_sec', 'rolling_rate_packet_min', 
              'header_length', 'payload_length']
    
processing = Processing(df=df_raw, 
                        arr=None,
                        columns_enc=columns_enc, 
                        columns_add=columns_add)
processing.process(normalize=True)
processing.df_process = processing.df_process.fillna(0)

# Save DataFrame
processing.df_process.to_csv(f"./DATA/PROCESS/df_process_{app}_GOOGLE_HOME.csv", index=False)
