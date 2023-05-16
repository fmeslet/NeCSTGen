
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""
FPORT = '1'


#######################
# FUNCTIONS / CLASS
#######################


# From : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
def standardize(x, min_x, max_x, a, b):
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
        #self.df_process['time_diff'] = self.df_raw['timestamps'].diff(1)
        #self.df_process['time_diff'] = self.df_process['time_diff'].fillna(0)
        # We set it to default array !
        #self.df_raw['time_diff'] = self.df_process['time_diff']
        
        for col in tqdm(self.columns_add):
            self.df_process[col] = standardize(
                  self.df_process[col], min_x=self.df_process[col].min(),
                   max_x=self.df_process[col].max(), a=0, b=1)
        #self.df_process = self.df_process.drop(['timestamps'], axis=1)
      
        #self.df_process["length_total"] = standardize(
        #      self.df_process["length_total"], min_x=self.df_process["length_total"].min(),
        #       max_x=self.df_process["length_total"].max(), a=0, b=1)
          
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

# If we load df_raw_*.csv files
df_raw = pd.read_csv(f"DATA/df_raw_LORA{FPORT}.csv")


for c in ['time_diff', 'rate', 'rolling_rate_byte_sec', 
          'rolling_rate_byte_min', 'rolling_rate_packet_sec', 
          'rolling_rate_packet_min']:

    df_raw[c] = np.log10(
        df_raw[c].values)

    inf_val = df_raw[c].iloc[1:].mean()
    
    df_raw.replace( # Imputation par la moyenne
        -np.inf, inf_val, inplace=True)

    
columns_enc = ['fport', 'mtype', 'code_rate', 'size', 'bandwidth',
               'spreading_factor', 'frequency', 'crc_status', 'gateway']
df_raw[columns_enc] = df_raw[columns_enc].astype(str)
columns_add = ['length_total', 'time_diff', 'snr', 'rssi', 
              'rate', 'rolling_rate_byte_sec', 'rolling_rate_byte_min',
               'rolling_rate_packet_sec', 'rolling_rate_packet_min',
                'header_length', 'payload_length', 'fcnt']
    
processing = Processing(df=df_raw, 
                        arr=None,
                        columns_enc=columns_enc,
                         columns_add=columns_add)
processing.process(normalize=True)
processing.df_process = processing.df_process.fillna(0)


processing.df_process.to_csv(f"./DATA/PROCESS/df_process_LORA{FPORT}.csv", index=False)

