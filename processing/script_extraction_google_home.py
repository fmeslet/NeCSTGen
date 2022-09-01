
#!/usr/bin/python3
#-*-coding: utf-8-*-

import scapy_layers

import gc
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, UDP, IP, Padding, Raw, load_layer, Ether, CookedLinux, PcapReader

load_layer("http")
load_layer("tls")

#FILENAME = "week3_friday.tcpdump"
DATA_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/DATA_RAW/GOOGLE_HOME/media_pcap_anonymized/"
#DATA_PATH = DATA_DIR + FILENAME
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/"



class Preprocessing():

  def __init__(self, lengths):
    self.df = pd.DataFrame() # set les columns avec les types et check la memoires.
    self.lengths = lengths
    self.filename = ""

  def reduce_memory_df(self, df):
    # Set type for each columns
    return df

  def add_df(self, feat_dict):
    layers_df = pd.DataFrame(feat_dict, index=[0])
    #print(layers_df)
    layers_df_reduced = self.reduce_memory_df(layers_df)
    self.df = pd.concat([self.df, layers_df_reduced], axis=0)

  def save_df(self, counter):
    self.df = self.df.reset_index(drop=True)
    self.df.to_csv(f"{SAVE_DIR}csv/df_{counter}_{self.filename}.csv", index=False)
    #self.df.to_csv(f"{SAVE_DIR}csv/df_{counter}.csv", index=False)
    self.df = pd.DataFrame()
    gc.collect()
    print(f"DataFrame {counter} saved, file {self.filename} !")

  def init_iterateur(self, iterator, value):
    i = 0
    while (i < value):
      iterator.next()
      i += 1

  def extract_features(self, packet, num_packet):
    feat_dict = {}

    # Extract layers
    layers = self.get_layers(packet=packet)
    for i in range(len(layers)):
      if(i >= len(layers)):
        feat_dict[f'layers_{i}'] = None
        feat_dict[f'length_{i}'] = int(0)
      else:
        feat_dict[f'layers_{i}'] = layers[i]
        feat_dict[f'length_{i}'] = int(len(packet[layers[i]]))

    # Extract timestamps
    feat_dict['timestamps'] = float(packet.time)

    # Extract total length
    feat_dict['length_total'] = int(len(packet))

    # Extract MAC address
    if ('Ether' in layers):
      feat_dict['mac_src'] = packet['Ethernet'].src
      feat_dict['mac_dst'] = packet['Ethernet'].dst
    elif ('Dot3' in layers):
      feat_dict['mac_src'] = packet['Dot3'].src
      feat_dict['mac_dst'] = packet['Dot3'].dst
    else:
      feat_dict['mac_src'] = None
      feat_dict['mac_dst'] = None
  
    # Extract IP address
    if ('IP' in layers):
      feat_dict['ip_src'] = packet['IP'].src
      feat_dict['ip_dst'] = packet['IP'].dst
    else:
      feat_dict['ip_src'] = None
      feat_dict['ip_dst'] = None

    # Extract flags if TCP and ports
    if ('TCP' in layers):
      feat_dict['flags'] = packet['TCP'].flags.value
      feat_dict['sport'] = packet['TCP'].sport
      feat_dict['dport'] = packet['TCP'].dport
    elif ('UDP' in layers):
      feat_dict['flags'] = None
      feat_dict['sport'] = packet['UDP'].sport
      feat_dict['dport'] = packet['UDP'].dport
    else:
      feat_dict['flags'] = None
      feat_dict['sport'] = None
      feat_dict['dport'] = None

    feat_dict['filename'] = self.filename
    feat_dict['num_packet'] = num_packet
    return feat_dict


  def fit(self, filename="", start=0, inter=1):
    self.filename = filename
    iterator = PcapReader(DATA_PATH)
    i = start # Packet counter
    j = 0 # Counter for saving

    # if(end is None):
    #   end = 9e9

    # Init iterateur to start value
    self.init_iterateur(iterator=iterator, value=start)
    print("Init done !")

    # while ((i < num_packets) and (i < end)):
    for pkt in iterator:

      # Extract features
      feat_dict = self.extract_features(pkt, num_packet=i)

      # Concat to  df
      self.add_df(feat_dict=feat_dict)

      # On incrémente le compteur 
      i += 1
      j += 1
      
      if(j == inter):
        self.save_df(counter=i)
        j = 0

    # Save the last data
    self.save_df(counter=i)


  def get_layers(self, packet):
    layer = []
    for i in packet.layers():
      name = str(i).split('.')[-1][:-2]
      layer.append(name)
    return layer

list_files = os.listdir(DATA_DIR)

for f in list_files:
  FILENAME = f
  DATA_PATH = DATA_DIR + FILENAME
  preprocessing = Preprocessing(lengths=1514*8)
  preprocessing.fit(filename=FILENAME, start=0, inter=2000) # 180000

# preprocessing = Preprocessing(lengths=1514*8)
# start=1001
# On génère à partir du 1001 paquets on compte à partir de 0 donc on a 0 à 999 en numéro
# preprocessing.fit(filename=FILENAME, start=0, inter=2000) # 180000
