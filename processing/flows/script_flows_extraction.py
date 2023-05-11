
#!/usr/bin/python3
#-*-coding: utf-8-*-

import scapy_layers

import gc
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, UDP, IP, Padding, Raw, load_layer, Ether, CookedLinux, PcapReader
from scapy.compat import bytes_encode

load_layer("http")
#load_layer("https")

FILENAME = "week3_friday"
APP_LIST = ['ARP', 'LLC', 'LOOP', 'SNAP', 'TELNET', 
            'HTTP', 'SSH', 'SNMP', 'SMTP', 'DNS', 
            'NTP', 'FTP', 'RIP', 'IRC', 'POP', 'ICMP', 
            'FINGER', 'TIME', "NETBIOS"]
MAIN_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
DATA_PATH = MAIN_DIR + FILENAME

#############################################
# Work on flow identified with IP and PORTS
#############################################

data_raw = pd.read_csv(f"{MAIN_DIR}df_{FILENAME}.csv")
#data_test = data_raw.copy()
columns = ['ip_src', 'ip_dst', 'sport', 'dport', 'applications']

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


# Fill NaN in Layers columns
data_raw.loc[data_raw['layers_2'].isna(), 'layers_2'] = "None"
data_raw.loc[data_raw['layers_3'].isna(), 'layers_3'] = "None"
data_raw.loc[data_raw['layers_4'].isna(), 'layers_4'] = "None"

data_raw.loc[data_raw['length_2'].isna(), 'length_2'] = 0
data_raw.loc[data_raw['length_3'].isna(), 'length_3'] = 0
data_raw.loc[data_raw['length_4'].isna(), 'length_4'] = 0

data_raw.loc[data_raw['ip_src'].isna(), 'ip_src'] = "None"
data_raw.loc[data_raw['ip_dst'].isna(), 'ip_dst'] = "None"
data_raw.loc[data_raw['flags'].isna(), 'flags'] = 0
data_raw.loc[data_raw['sport'].isna(), 'sport'] = 0
data_raw.loc[data_raw['dport'].isna(), 'dport'] = 0

try:
    apply_http_corrections(data_raw)
except:
    pass

set_applications(data_raw)


#data_test = data_raw[data_raw['applications'] == 'HTTP'].reset_index(drop=True)
data_test = data_raw.copy()

sessions = data_test.groupby( #[condition]
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions[(sessions['sport'] != 0) & (sessions['dport'] != 0)].copy()
#sessions = sessions[(sessions['sport'] < 1024)].reset_index(drop=True)
# MAUVAISE IDEES POUR UN FLUX IRC

for i in range(sessions.shape[0]): #result.shape[0]
    condition_flow = (((data_test['ip_src'] == sessions['ip_src'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_dst'].iloc[i]) & 
                     (data_test['sport'] == sessions['sport'].iloc[i]) &
                     (data_test['dport'] == sessions['dport'].iloc[i]) &
                     (data_test['applications'] == sessions['applications'].iloc[i])) | 
                    ((data_test['ip_src'] == sessions['ip_dst'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_src'].iloc[i]) & 
                     (data_test['sport'] == sessions['dport'].iloc[i]) &
                     (data_test['dport'] == sessions['sport'].iloc[i]) &
                     (data_test['applications'] == sessions['applications'].iloc[i])))
    # Add flow id
    data_test.loc[condition_flow, "flow_id"] = i

    if(data_test[condition_flow].shape[0] <= 1):
        print(sessions.iloc[i])
    
flows = data_test[
    'flow_id'].value_counts().sort_index().index.values

# Reasign ordered flow
#for i, flow_id in enumerate(flows):
#    condition_flow_id = (data_test['flow_id'] == flow_id)
#    data_test.loc[condition_flow_id, "flow_id"] = i

data_test.to_csv(f"{MAIN_DIR}df_{FILENAME}_flows_tmp.csv", index=False)

#########################################
# Work on pseudo flow with no IP layer...
##########################################

# ARP est à part!
condition_arp = ((data_raw['sport'] == 0) & 
                 (data_raw['dport'] == 0) & 
                 (data_raw['layers_1'] == 'ARP'))

# LLC layers_1 Raw layers_2
condition_llc_raw = (((data_raw['sport'] == 0) & 
                     (data_raw['dport'] == 0) & 
                     (data_raw['layers_1'] == 'LLC') &
                     (data_raw['layers_2'] == 'Raw')))

# Les reply CISCO
condition_ether_raw = ((data_raw['sport'] == 0) & 
                     (data_raw['dport'] == 0) & 
                     (data_raw['layers_0'] == 'Ether') &
                     (data_raw['layers_1'] == 'Raw') &
                     (data_raw['layers_2'] == 'None'))

# LLC SNAP
condition_llc_snap = ((data_raw['sport'] == 0) & 
                      (data_raw['dport'] == 0) & 
                      (data_raw['layers_1'] == 'LLC') &
                      (data_raw['layers_2'] == 'SNAP'))

# ICMP
condition_icmp = ((data_raw['sport'] == 0) &
                  (data_raw['dport'] == 0) &
                  (data_raw['layers_1'] == 'IP') &
                  (data_raw['layers_2'] == 'ICMP'))

start_index = int(data_test['flow_id'].max() + 1)
columns = ['mac_src', 'mac_dst', 'layers_1', 'applications']

for cond in [condition_ether_raw, condition_arp,
             condition_llc_raw, condition_llc_snap, 
             condition_icmp]:
    sessions = data_raw[cond].groupby( #[condition]
        columns).size().reset_index().rename(
        columns={0:'count'})

    for i in range(sessions.shape[0]): #result.shape[0]
        condition_flow = (((data_raw['mac_src'] == sessions['mac_src'].iloc[i]) &
                          (data_raw['mac_dst'] == sessions['mac_dst'].iloc[i]) &
                          (data_raw['layers_1'] == sessions['layers_1'].iloc[i]) & 
                          (data_raw['applications'] == sessions['applications'].iloc[i]) &
                          (data_raw['sport'] == 0) & (data_raw['dport'] == 0)) |
                        ((data_raw['mac_src'] == sessions['mac_dst'].iloc[i]) &
                         (data_raw['mac_dst'] == sessions['mac_src'].iloc[i]) &
                         (data_raw['layers_1'] == sessions['layers_1'].iloc[i]) & 
                         (data_raw['applications'] == sessions['applications'].iloc[i]) &
                         (data_raw['sport'] == 0) & (data_raw['dport'] == 0)))
        
        # Add flow id
        data_test.loc[condition_flow, "flow_id"] = i+start_index

    start_index = start_index+sessions.shape[0]
    
#start_index = int(data_test['flow_id'].max()+1)
#flows = data_test[
#    'flow_id'].value_counts().sort_index().index.values[
#    start_index:]

# Reasign ordered flow
#for i, flow_id in enumerate(flows):
#    condition_flow_id = (data_test['flow_id'] == flow_id)
#    data_test.loc[condition_flow_id, "flow_id"] = i+start_index
    
data_test.to_csv(f"{MAIN_DIR}df_{FILENAME}_flows.csv", index=False)

