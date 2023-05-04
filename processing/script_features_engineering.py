
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""
DATA_DIR_REF = "/users/rezia/fmesletm/DATA_GENERATION/DATA/DATA_RAW/GOOGLE_HOME/media_pcap_anonymized/"
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/"


#######################
# EXTRACT NEW FEATURES
#######################


def load_data():
    
    df_week1_monday = pd.read_csv(f"{MAIN_DIR}df_week1_monday_flows.csv")
    df_week1_monday['day'] = 'monday'
    df_week1_monday['week'] = str(1)
    df_week1_tuesday = pd.read_csv(f"{MAIN_DIR}df_week1_tuesday_flows.csv")
    df_week1_tuesday['day'] = 'tuesday'
    df_week1_tuesday['week'] = str(1)
    df_week1_wednesday = pd.read_csv(f"{MAIN_DIR}df_week1_wednesday_flows.csv")
    df_week1_wednesday['day'] = 'wednesday'
    df_week1_wednesday['week'] = str(1)
    df_week1_thursday = pd.read_csv(f"{MAIN_DIR}df_week1_thursday_flows.csv")
    df_week1_thursday['day'] = 'thursday'
    df_week1_thursday['week'] = str(1)
    df_week1_friday = pd.read_csv(f"{MAIN_DIR}df_week1_friday_flows.csv")
    df_week1_friday['day'] = 'friday'
    df_week1_friday['week'] = str(1)
    df_week3_monday = pd.read_csv(f"{MAIN_DIR}df_week3_monday_flows.csv")
    df_week3_monday['day'] = 'monday'
    df_week3_monday['week'] = str(3)
    df_week3_tuesday = pd.read_csv(f"{MAIN_DIR}df_week3_tuesday_flows.csv")
    df_week3_tuesday['day'] = 'tuesday'
    df_week3_tuesday['week'] = str(3)
    df_week3_wednesday = pd.read_csv(f"{MAIN_DIR}df_week3_wednesday_flows.csv")
    df_week3_wednesday['day'] = 'wednesday'
    df_week3_wednesday['week'] = str(3)
    df_week3_thursday = pd.read_csv(f"{MAIN_DIR}df_week3_thursday_flows.csv")
    df_week3_thursday['day'] = 'thursday'
    df_week3_thursday['week'] = str(3)
    df_week3_friday = pd.read_csv(f"{MAIN_DIR}df_week3_friday_flows.csv")
    df_week3_friday['day'] = 'friday'
    df_week3_friday['week'] = str(3)
    
    dfs = [df_week1_monday, df_week1_tuesday, 
           df_week1_wednesday, df_week1_thursday, 
           df_week1_friday, df_week3_monday, df_week3_tuesday, 
           df_week3_wednesday, df_week3_thursday, 
           df_week3_friday]
    
    # CHANGE ARR
    dfs_new = []
    for df, arr, i in zip(dfs, arrs, range(len(dfs))):
        cond_app = (df['applications'] == 'SNMP')
        indexes = df[cond_app].index.values
        df_tmp = df[cond_app].reset_index(drop=True)
        dfs_new.append(df_tmp)
    
    
    # CHANGE DF
   # for i, df in enumerate(dfs):
    #    cond_app = (df['applications'] == 'SNMP')
    #    df_tmp = df[cond_app].reset_index(drop=True)
    #    dfs[i] = df_tmp
    df = pd.concat(dfs_new, axis=0)
    
    #df = df_week1_monday
    #arr = arr_week1_monday
    
    #condition_ports = ((df['sport'] == 80) | 
    #                   (df['dport'] == 80))
    #arr = arr[df[condition_ports].index.values]
    #df = df[condition_ports].reset_index(drop=True)
   
    
    # Fill NaN in Layers columns
    df.loc[df['layers_2'].isna(), 'layers_2'] = "None"
    df.loc[df['layers_3'].isna(), 'layers_3'] = "None" 
    df.loc[df['layers_4'].isna(), 'layers_4'] = "None" 
    df['layers_5'] = "None"
    #df.loc[df['layers_5'].isna(), 'layers_5'] = "None"

    df.loc[df['length_2'].isna(), 'length_2'] = 0
    df.loc[df['length_3'].isna(), 'length_3'] = 0
    df.loc[df['length_4'].isna(), 'length_4'] = 0
    df['length_5'] = 0
    #df.loc[df['length_5'].isna(), 'length_5'] = 0

    df.loc[df['ip_src'].isna(), 'ip_src'] = "None"
    df.loc[df['ip_dst'].isna(), 'ip_dst'] = "None" 
    df.loc[df['flags'].isna(), 'flags'] = 0 
    df.loc[df['sport'].isna(), 'sport'] = 0
    df.loc[df['dport'].isna(), 'dport'] = 0
    
    print("# Apply HTTP corrections")
    try:
        apply_http_corrections(df)
    except:
        pass
    
    print("# Set applications")
    set_applications(df)
    
    # Extract day and week
    def some_func(a, b):
        #print(dict_pad[a])
        return str(a)+'_'+str(b)

    df['day_week'] = df[['day', 'week']].apply(
        lambda x: some_func(a=x['day'], b=x['week']), axis=1)
    
    # Pour chaque jour et semaine 
    day_week_values = df['day_week'].value_counts().index.values
    df = df.sort_values(by=['timestamps'])
    df = df.reset_index(drop=True)
    
    for day_week in day_week_values:
        
        for app in ['SNMP']:#APP_LIST:
            print(f"{app} in progress")
            condition_app = ((df['applications'] == app) & (df['day_week'] == day_week))
            print("df[condition_app].shape : ", df[condition_app].shape)

            # Compute time diff
            df.loc[condition_app, 'time_diff'] = df[
                condition_app]['timestamps'].diff(1).fillna(0)
            df.loc[condition_app, 'rate'] = 0
            df.loc[df[condition_app].index.values[1:], 'rate'] = df[
                condition_app]["length_total"].values[:-1] / df[
                condition_app]["time_diff"].values[1:]
            #df[condition_app].loc[1:, 'rate'] = df[condition_app]["length_total"].values[:-1] / df[
            #    condition_app]["time_diff"].values[1:]

            def map_time(timestamp):
                local_time = time.localtime(timestamp)
                return pd.Timestamp(pd.to_datetime(
                                    time.strftime('%Y%d%m %H:%M:%S', local_time), 
                                    format='%Y%d%m %H:%M:%S'))

            df_rate = df[condition_app][['timestamps', 'length_total']].copy()
            #.reset_index(drop=True)#.copy()
            df_rate['datetime'] = df_rate['timestamps'].map(map_time)
            df_rate = df_rate.set_index('datetime')#.sort_index()


            df.loc[condition_app, "rolling_rate_byte_sec"] = df_rate['length_total'].rolling(
                '1s').sum().values#.reset_index(drop=True)
            df.loc[condition_app, "rolling_rate_byte_min"] = df_rate['length_total'].rolling(
                '60s').sum().values#.reset_index(drop=True)

            df.loc[condition_app, "rolling_rate_packet_sec"] = df_rate['length_total'].rolling(
                '1s').count().values#.reset_index(drop=True)
            df.loc[condition_app, "rolling_rate_packet_min"] = df_rate['length_total'].rolling(
                '60s').count().values#.reset_index(drop=True) 

        # GET LENGTHS
        for i in reversed(range(0, 6)):
            #print(f"Layers {i}")
            condition = (df[f"layers_{i}"] == "None")
            df.loc[condition, "payload_layers"] = i
            if(i > 0):
                df.loc[condition, "header_length"] = df[
                        condition]["length_total"] - df[condition][f"length_{i-1}"]
        df["payload_length"] = df["length_total"] - df["header_length"]
    
    return arr, df



#######################
# EXTRACT NEW FEATURES
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
            self.df_process[col] = functions.standardize(
              self.df_process[col], min_x=self.df_process[col].min(),
                max_x=self.df_process[col].max(), a=0, b=1)
            
        return le

    
    def process(self, normalize=True):
        #self.df_process['time_diff'] = self.df_raw['timestamps'].diff(1)
        #self.df_process['time_diff'] = self.df_process['time_diff'].fillna(0)
        # We set it to default array !
        #self.df_raw['time_diff'] = self.df_process['time_diff']
        
        for col in tqdm(self.columns_add):
            self.df_process[col] = functions.standardize(
                  self.df_process[col], min_x=self.df_process[col].min(),
                   max_x=self.df_process[col].max(), a=0, b=1)
        #self.df_process = self.df_process.drop(['timestamps'], axis=1)
      
        #self.df_process["length_total"] = functions.standardize(
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
            df_new[col] = functions.standardize(
                       df_copy[col], min_x=0,
                       max_x=1, a=0,
                       b=len(self.dict_le[col].classes_)-1)

        # For columns added
        for col in self.columns_add:    
            df_new[col] = functions.standardize(
                       df_copy[col], min_x=0,
                       max_x=1, a=self.df_raw[col].min(),
                       b=self.df_raw[col].max())

        return df_new



for c in ['time_diff', 'rate', 'rolling_rate_byte_sec', 
              'rolling_rate_byte_min', 'rolling_rate_packet_sec', 
              'rolling_rate_packet_min']:

    df_raw[c] = np.log10(
        df_raw[c].values)

    inf_val = df_raw[c].iloc[1:].mean()
    
    df_raw.replace( # Imputation par la moyenne
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
                        arr=None,#arr_raw,
                        columns_enc=columns_enc, # , 
                         columns_add=columns_add)
processing.process(normalize=True)
processing.df_process = processing.df_process.fillna(0)




list_files = os.listdir(DATA_DIR_REF)
df_result = pd.DataFrame()

for f in list_files:
  FILENAME = f

  df_result_tmp = aggregate_df_file(filename=FILENAME, path=DATA_DIR, file_type='.csv')
  num = df_result_tmp['num_packet'].values.max() + 1

  df_result_tmp = df_result_tmp.sort_values(by='num_packet', ascending=True)
  df_result_tmp = df_result_tmp.reset_index(drop=True)
 
  df_result = pd.concat([df_result, df_result_tmp], axis=0)

df_result.to_csv(f"{SAVE_DIR}df_{FILENAME_SAVE}.csv", index=False)
