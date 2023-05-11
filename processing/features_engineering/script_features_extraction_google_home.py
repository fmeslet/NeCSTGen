
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""
FILENAME_SAVE = "media_pcap_anonymized"

DATA_DIR_REF = "/users/rezia/fmesletm/DATA_GENERATION/DATA/DATA_RAW/GOOGLE_HOME/media_pcap_anonymized/"
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/"

PROTO = 'TCP' # or UDP


#######################
# FUNCTIONS
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
        np.array: Data standardized.
    """
    # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new


def set_applications(data):

    cond_arp = ((data['sport'] == 0) &
                 (data['dport'] == 0) &
                 (data['layers_1'] == 'ARP'))

    cond_llc = (((data['sport'] == 0) &
                  (data['dport'] == 0) &
                  (data['layers_1'] == 'LLC') &
                  (data['layers_2'] == 'Raw')))

    cond_loop = ((data['sport'] == 0) &
                 (data['dport'] == 0) &
                 (data['layers_0'] == 'Ether') &
                 (data['layers_1'] == 'Raw') &
                 (data['layers_2'] == 'None'))

    cond_snap = ((data['sport'] == 0) &
                  (data['dport'] == 0) &
                  (data['layers_1'] == 'LLC') &
                  (data['layers_2'] == 'SNAP'))

    cond_telnet = ((data['sport'] == 23) | 
                   (data['dport'] == 23) |
                   (data['layers_3'] == 'TELNET'))
    cond_http = ((data['sport'] == 80) | 
                 (data['dport'] == 80))
    cond_ssh = ((data['layers_3'] == 'SSH') |
                ((data['sport'] == 22) | 
                (data['dport'] == 22)))
    cond_snmp = (data['layers_3'] == 'SNMP')
    cond_smtp = (((data['layers_3'] == 'SMTPRequest') | 
                 (data['layers_3'] == 'SMTPResponse')) |
                 ((data['sport'] == 25) | 
                (data['dport'] == 25)))
    cond_dns = ((data['sport'] == 53) | 
                (data['dport'] == 53) |
                (data['layers_3'] == 'DNS')) 
    cond_ntp = (data['layers_3'] == 'NTPHeader') 
    cond_ftp = (((data['layers_3'] == 'FTPRequest') | 
                (data['layers_3'] == 'FTPResponse')) |
                ((data['sport'] == 20) | 
                (data['dport'] == 20)) |
                ((data['sport'] == 21) | 
                (data['dport'] == 21)))
    cond_rip = (data['layers_3'] == 'RIP')
    cond_irc = (((data['layers_3'] == 'IRCRes') | 
                (data['layers_3'] == 'IRCReq')) | 
                ((data['sport'] == 113) | 
                (data['dport'] == 113)) | 
                ((data['sport'] == 6667) | 
                (data['dport'] == 6667)) )
    cond_pop = ((data['layers_3'] == 'POP') | 
                ((data['sport'] == 110) | 
                (data['dport'] == 110)))
    cond_icmp = (data['layers_2'] == 'ICMP')
    cond_finger = ((data['sport'] == 79) | 
                   (data['dport'] == 79))
    cond_time = ((data['sport'] == 37) | 
                 (data['dport'] == 37))
    cond_netbios = ((data['sport'] == 137) | 
                   (data['dport'] == 137))

    conditions = [cond_arp, cond_llc, cond_loop, 
                  cond_snap, cond_telnet, cond_http,
                  cond_ssh, cond_snmp, cond_smtp,
                  cond_dns, cond_ntp, cond_ftp,
                  cond_rip, cond_irc, cond_pop,
                  cond_icmp, cond_finger, cond_time, 
                  cond_netbios]

    for cond, app in zip(conditions, APP_LIST):
        data.loc[cond, 'applications'] = app
    
    # Correction for label encoding
    data['applications'] = data['applications'].astype(str)


def load_data():

    df = pd.read_csv(f"{SAVE_DIR}df_{FILENAME_SAVE}.csv")
    
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
    #apply_http_corrections(df)
    
    #print("# Set applications")
    set_applications(df)
    
    df = df.sort_values(by=['timestamps'])
    
    filenames = df['filename'].value_counts().index.values
    for f in tqdm(filenames):
        for app in [PROTO]: # or TCP
            #print(f"{app} in progress")
            #condition_app = (df['filename'] == f)
            condition_app = ((df['filename'] == f) & 
                             (df['layers_2'] == app))

            #print("Shape : ", df[condition_app].shape)

            # Compute time diff
            df.loc[condition_app, 'time_diff'] = df[condition_app]['timestamps'].diff(1).fillna(0)
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

            #print("df_rate shape : ", df_rate.shape)
            #print("df_rate head : ", df_rate.head())
            #print("df_rate tail : ", df_rate.tail())

            #df["rate_byte_sec"] = df_rate['length_total'].resample(
            #    '1s').sum().reset_index(drop=True)

            df.loc[condition_app, "rolling_rate_byte_sec"] = df_rate['length_total'].rolling(
                '1s').sum().values#.reset_index(drop=True)
            df.loc[condition_app, "rolling_rate_byte_min"] = df_rate['length_total'].rolling(
                '60s').sum().values#.reset_index(drop=True)

            #print("rolling 1s : ", df_rate['length_total'].rolling(
            #    '1s').sum().values)
            #print("attribution : ", df.loc[condition_app]["rolling_rate_byte_sec"])

            #print(df[condition_app]["rolling_rate_byte_sec"].head())

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
    
    #condition_ports = ((df['sport'] == 25) | 
    #                   (df['dport'] == 25))
    #arr = arr[df[condition_ports].index.values]
    #df = df[condition_ports].reset_index(drop=True)
    
    return df



#######################
# MAIN
#######################



df_raw = load_data()
df_raw.to_csv(
    "/users/rezia/fmesletm/DATA_GENERATION/DATA/PROCESS/df_raw_{PROTO}_GOOGLE_HOME.csv", index=False)


