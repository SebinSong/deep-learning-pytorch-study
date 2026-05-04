import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()
data_fpath = resolve_path('../data/winequality-red.csv')

F = nn.functional

# Util functions
def draw_box_plot(df):
  plt.figure(figsize=(15, 8))
  sns.boxplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

def draw_violin_plot(df):
  plt.figure(figsize=(15, 8))
  sns.violinplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

z_score_df = lambda df: df.apply(stats.zscore)

df = pd.read_csv(data_fpath, sep=';', header=0)

# filter and normalize the data
df = df[df['total sulfur dioxide'] < 200] # 1. remove the outliers
df_norm = z_score_df(df)

draw_box_plot(df_norm)
