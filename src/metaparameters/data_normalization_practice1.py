import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import seaborn as sns
import scipy.stats as stats

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

iris_subdf = iris[iris.columns[:-1]]
z_score_df = lambda df: df.apply(stats.zscore)

def draw_violin_plot(df, zscore = False): 
  if zscore:
    df = z_score_df(df)

  plt.figure(figsize=(10, 6))
  sns.violinplot(data=df)
  plt.xticks(rotation=20)
  plt.show()

draw_violin_plot(iris_subdf, True)
