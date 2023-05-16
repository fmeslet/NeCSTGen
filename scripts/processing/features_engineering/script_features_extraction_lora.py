
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""


#######################
# FUNCTIONS
#######################


def get_dynamic(df):
    
    ref = df[by].value_counts().index.values
    
    for g in ref:
        print(f"{g} in progress")
        condition_gateway = (df[by] == g)
    
        # Compute time diff
        df.loc[condition_gateway, 'time_diff'] = \
            df[condition_gateway]['timestamps'].diff(1).fillna(0)
        df.loc[condition_gateway, 'rate'] = 0
        df.loc[df[condition_gateway].index.values[1:], 'rate'] = df[
            condition_gateway]["length_total"].values[:-1] / df[
            condition_gateway]["time_diff"].values[1:]

        def map_time(timestamp):
            local_time = time.localtime(timestamp)
            return pd.Timestamp(pd.to_datetime(
                                time.strftime('%Y%d%m %H:%M:%S', local_time), 
                                format='%Y%d%m %H:%M:%S'))

        df_rate = df[condition_gateway][['timestamps', 'length_total']].copy()
        df_rate['datetime'] = df_rate['timestamps'].map(map_time)
        df_rate = df_rate.set_index('datetime')


        df.loc[condition_gateway, "rolling_rate_byte_sec"] = df_rate['length_total'].rolling(
            '1s').sum().values
        df.loc[condition_gateway, "rolling_rate_byte_min"] = df_rate['length_total'].rolling(
            '60s').sum().values

        df.loc[condition_gateway, "rolling_rate_packet_sec"] = df_rate['length_total'].rolling(
            '1s').count().values
        df.loc[condition_gateway, "rolling_rate_packet_min"] = df_rate['length_total'].rolling(
            '60s').count().values
        
    return df


def load_data():
    df  = pd.DataFrame()
    
    for f in os.listdir(f'{MAIN_DIR}'):

        df_tmp = pd.read_csv(f'{MAIN_DIR}{f}', encoding="utf8")
        
        # Extract some features
        df_tmp['payload_length'] = df_tmp['physical_payload'].map(lambda x : len(str(x)))
            
        # MTYPE + RFU + Major + ID + FC + FCounter + FP
        df_tmp['header_length'] = 9
        df_tmp['length_total'] = df_tmp['payload_length'] + df_tmp['header_length']

        df_tmp['time'] = pd.to_datetime(df_tmp['time'])
        df_tmp['timestamps'] = df_tmp['time'].astype(int) / 10**9

        #df_tmp = df_tmp.sort_values(by=['timestamps'])
        df = pd.concat([df, df_tmp], axis=0)
     
    # Remove NAN values
    df['physical_payload'] = df['physical_payload'].fillna("")
    df['fport'] = df['fport'].fillna(-2)
    df['bandwidth'] = df['bandwidth'].fillna(-1)
    df['code_rate'] = df['code_rate'].fillna(-1)
    df['spreading_factor'] = df['spreading_factor'].fillna(-1)
        
    return df



#######################
# MAIN
#######################


# Load data
data_raw = load_data()

# Select a gateway and a fport
cond = ((data_raw['gateway'] == '0000024b0b031c97') &
        (data_raw['fport'] == 10))
data_raw_tmp = data_raw[cond].reset_index(drop=True)
data_raw_tmp = data_raw_tmp.sort_values(by=['timestamps']).reset_index(drop=True)
df_raw = get_dynamic(data_raw_tmp)

# Save data
# LORA_1 => for fport = 1
# LORA_10 => for fport = 10
df_raw.to_csv(f"DATA/PROCESS/df_raw_LORA_1_.csv", index=False)
