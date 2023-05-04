
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os

FILENAME = "week3_friday.tcpdump"

DATA_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/csv/"
DATA_PATH = "/users/rezia/fmesletm/DATA_GENERATION/DATA/"
SAVE_DIR = "/users/rezia/fmesletm/DATA_GENERATION/DATA/" 

# Aggregate file
def aggregate_df_file(filename, path, file_type='.txt'):
  """Aggregate the DataFrame generated.

  Args:
      filename (str): File with filename inside to load and concat with others.
      path (str): Folder to load the file named filename.
      file_type (str, optional): Filter for the file extension. Defaults to '.txt'.

  Returns:
      pandas.DataFrame: DataFrame with the DataFrame concatenate.
  """
  df = pd.DataFrame()
  for f in os.listdir(path):
    print(f)
    if ((filename in f) and (file_type in f)):
      df_tmp = pd.read_csv(path + f)
      df = pd.concat([df, df_tmp], axis=0)
      #df = df.reset_index(drop=True)
  return df

df_result = aggregate_df_file(filename=FILENAME, path=DATA_DIR, file_type='.csv')
num = df_result['num_packet'].values.max() + 1

df_result = df_result.sort_values(by='num_packet', ascending=True)
df_result = df_result.reset_index(drop=True)
df_result.to_csv(f"{SAVE_DIR}df_{num}_{FILENAME}.csv", index=False)

# Supprimer les arrays traités
