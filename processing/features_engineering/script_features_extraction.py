
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os


MAIN_DIR = ""
APP_LIST = ['ARP', 'LLC', 'LOOP', 'SNAP', 'TELNET', 
            'HTTP', 'SSH', 'SNMP', 'SMTP', 'DNS', 
            'NTP', 'FTP', 'RIP', 'IRC', 'POP', 'ICMP', 
            'FINGER', 'TIME', "NETBIOS"]
APP = 'SMTP'


#######################
# FUNCTIONS
#######################


def apply_http_corrections(data):
    # Apply PPTP
    condition_pptp = (data['layers_3'] == "PPTP") & (data['sport'] == 80)
    index = data[condition_pptp].index.values
    data.iloc[index[0], data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index[0], data.columns.get_loc('layers_4')] =  'HTTPResponse'
    data.iloc[index[1], data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index[1], data.columns.get_loc('layers_4')] =  'Raw'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']

    # Apply PPTP
    condition_pptp = (data['layers_3'] == "PPTP") & (data['dport'] == 80)
    index = data[condition_pptp].index.values
    data.iloc[index[0], data.columns.get_loc('layers_3')] =  'Padding'

    # Apply IRC Response correction
    condition_irc_res = (data['layers_3'] == "IRCRes") & (data['dport'] == 80)
    index = data[condition_irc_res].index.values
    data.iloc[index, data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index, data.columns.get_loc('layers_4')] =  'HTTPRequest'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']

    # Apply IRC Req correction
    condition_irc_req = (data['layers_3'] == "IRCReq") & (data['sport'] == 80)
    index = data[condition_irc_req].index.values
    data.iloc[index, data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index, data.columns.get_loc('layers_4')] =  'Raw'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']

    # Apply H2Frame correction
    condition_h2frame = (data['layers_3'] == "H2Frame") & (data['sport'] == 80)
    index = data[condition_h2frame].index.values
    data.iloc[index, data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index, data.columns.get_loc('layers_4')] =  'Raw'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']

    # Apply Raw correction
    condition_raw = ((data['layers_3'] == "Raw") & 
                     (data['sport'] == 80) & 
                     (data['length_0'] >= 1078))
    index = data[condition_raw].index.values
    data.iloc[index, data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index, data.columns.get_loc('layers_4')] =  'Raw'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']

    condition_raw = ((data['layers_3'] == "Raw") & 
                     (data['sport'] == 80) & 
                     (data['length_0'] < 1078))
    index = data[condition_raw].index.values
    data.iloc[index, data.columns.get_loc('layers_3')] =  'HTTP'
    data.iloc[index, data.columns.get_loc('layers_4')] =  'HTTPResponse'
    data.iloc[index, data.columns.get_loc('length_4')] = data.iloc[index]['length_3']


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
    

    # All week and all day are not usefull
    # we can select only one day or one week
    # depending on the number of packet we want
    # and the generation time range we want to use.
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
    
    # Keep only the targeted application
    dfs_new = []
    for df, arr, i in zip(dfs, arrs, range(len(dfs))):
        cond_app = (df['applications'] == APP)
        indexes = df[cond_app].index.values
        df_tmp = df[cond_app].reset_index(drop=True)
        dfs_new.append(df_tmp)
    
    df = pd.concat(dfs_new, axis=0)
    
    #condition_ports = ((df['sport'] == 80) | 
    #                   (df['dport'] == 80))
    #df = df[condition_ports].reset_index(drop=True)
   
    # Fill NaN in Layers columns
    df.loc[df['layers_2'].isna(), 'layers_2'] = "None"
    df.loc[df['layers_3'].isna(), 'layers_3'] = "None" 
    df.loc[df['layers_4'].isna(), 'layers_4'] = "None" 
    df['layers_5'] = "None"

    df.loc[df['length_2'].isna(), 'length_2'] = 0
    df.loc[df['length_3'].isna(), 'length_3'] = 0
    df.loc[df['length_4'].isna(), 'length_4'] = 0
    df['length_5'] = 0

    df.loc[df['ip_src'].isna(), 'ip_src'] = "None"
    df.loc[df['ip_dst'].isna(), 'ip_dst'] = "None" 
    df.loc[df['flags'].isna(), 'flags'] = 0 
    df.loc[df['sport'].isna(), 'sport'] = 0
    df.loc[df['dport'].isna(), 'dport'] = 0
    
    # Only in case of HTTP application
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
        
        for app in [APP]: # or APP_LIST:
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
    
    return df



#######################
# CODE
#######################


df_raw = load_data()

# Select day or week we want to 
# keep for the modelling
# cond = ((df_raw['day'] == 'wednesday') &
#         (df_raw['week'] == "1"))
# df_raw = df_raw[cond].reset_index(drop=True)

df_raw.to_csv(f"DATA/PROCESS/df_raw_{APP}.csv", index=False)
