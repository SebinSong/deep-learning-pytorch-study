import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/heart_disease_uci.csv').resolve()

df = pd.read_csv(data_fpath, header=0, sep=',')
mode_df = df.mode()

def sanitize_data():
  global df
  # 1. Drop irrelevant columns ('id', 'dataset')
  df = df.drop(['id', 'dataset'], axis=1)

  # 2. replace all NaN in dataframe with either mode or median
  #    mode: for categorical column
  #    median: for numerical column (why not using mean? - To use a representative value that is less sensitive to outliers)
  all_columns = df.columns
  categorical_cols = []
  numerical_cols = []

  for col in all_columns:
    column = df[col]
    if column.dtype == object:
      categorical_cols.append(col)
      df[col] = column.fillna(column.mode()[0])
    else:
      numerical_cols.append(col)
      df[col] = column.fillna(column.median())

  # 3. Convert all categorical values to their numeric equivalent
  cols_to_one_hot_encode = ['cp', 'restecg']
  cols_bool = ['fbs', 'exang']
  col_mappings = {
    'sex': { 'Male': 0, 'Female': 1 },
    'slope': { 'downsloping': -1, 'flat': 0, 'upsloping': 1 },
    'thal': { 'reversable defect': -1, 'normal': 0, 'fixed defect': 1 }
  }

  # 3-1. perform one-hot encoding for some columns
  df = pd.get_dummies(df, columns=cols_to_one_hot_encode, dtype=float)

  # 3-2. Turn boolean Series into float / apply mapping to each columns
  for col in cols_bool:
    df[col] = df[col].astype(float)

  for col_name in col_mappings.keys():
    column = df[col_name]
    df[col_name] = column.map(col_mappings[col_name]).astype(float)

  # TODO:
  # normalize the data by z-scoring them. (!IMPORTANT! - Do not scale 'num' column which will be used as targets)
  # split the data and labels and generate train/test data loaders

sanitize_data()
