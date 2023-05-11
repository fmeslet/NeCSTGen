
#!/usr/bin/python3
#-*-coding: utf-8-*-

import gc
import pandas as pd
import numpy as np


FILENAME = "UDP_GOOGLE_HOME"
APP_LIST = ['ARP', 'LLC', 'LOOP', 'SNAP', 'TELNET', 
            'HTTP', 'SSH', 'SNMP', 'SMTP', 'DNS', 
            'NTP', 'FTP', 'RIP', 'IRC', 'POP', 'ICMP', 
            'FINGER', 'TIME', "NETBIOS"]
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/PROCESS/"
DATA_PATH = MAIN_DIR + FILENAME

#############################################
# Work on flow identified with IP and PORTS
#############################################

data_raw = pd.read_csv(f"DATA/GOOGLE_HOME/media_pcap_anonymized/df_results.csv")
columns = ['ip_src', 'ip_dst', 'sport', 'dport', 'layers_2', 'filename']

#data_raw = pd.read_csv(f"{MAIN_DIR}df_raw_{FILENAME}.csv")
#columns = ['ip_src', 'ip_dst', 'sport', 'dport', 'applications']


data_test = data_raw.copy()

sessions = data_test.groupby(
    columns).size().reset_index().rename(
    columns={0:'count'})

for i in range(sessions.shape[0]):
    condition_flow = (((data_test['ip_src'] == sessions['ip_src'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_dst'].iloc[i]) & 
                     (data_test['sport'] == sessions['sport'].iloc[i]) &
                     (data_test['dport'] == sessions['dport'].iloc[i]) &
                     (data_test['layers_2'] == sessions['layers_2'].iloc[i]) &
                     (data_test['filename'] == sessions['filename'].iloc[i])) |
                    ((data_test['ip_src'] == sessions['ip_dst'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_src'].iloc[i]) & 
                     (data_test['sport'] == sessions['dport'].iloc[i]) &
                     (data_test['dport'] == sessions['sport'].iloc[i]) &
                     (data_test['layers_2'] == sessions['layers_2'].iloc[i]) &
                     (data_test['filename'] == sessions['filename'].iloc[i])))

    # Add flow id
    data_test.loc[condition_flow, "flow_id"] = i

    if(data_test[condition_flow].shape[0] <= 1):
        print(sessions.iloc[i])
    
flows = data_test[
    'flow_id'].value_counts().sort_index().index.values

data_test.to_csv(f"DATA/GOOGLE_HOME/media_pcap_anonymized/df_results_flows.csv", index=False)
#data_test.to_csv(f"{MAIN_DIR}df_{FILENAME}_flows_tmp.csv", index=False)
