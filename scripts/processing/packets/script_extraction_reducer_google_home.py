
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os

FILENAME_SAVE = "media_pcap_anonymized"

DATA_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/csv/"
DATA_DIR_REF = "/users/rezia/fmesletm/DATA_GENERATION/DATA/DATA_RAW/GOOGLE_HOME/media_pcap_anonymized/"
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/GOOGLE_HOME/media_pcap_anonymized/"

# Aggregate file
def aggregate_df_file(filename, path, file_type='.txt'):
  df = pd.DataFrame()
  for f in os.listdir(path):
    print(f)
    if ((filename in f) and (file_type in f)):
      df_tmp = pd.read_csv(path + f)
      df = pd.concat([df, df_tmp], axis=0)
      #df = df.reset_index(drop=True)
  return df


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
