import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.stats as stats
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
from sebin_utils import draw_box_plot, to_numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Sanitize the raw-data from training

# 1. drop 'quality' column - Don't need it in predicting the residual sugar
data.drop(columns=['quality'], inplace=True)

# 2. drop some outliers
data = data[data['total sulfur dioxide'] < 200]

# 3. Normalize the data
# - using 'StandardScaler' here to perform z-scoring on the data and then
#   reverse transform the predicted data later (it remembers the mean/std info of the fitted data)
#
# some terminologies:
#.  1) fitting -> Computing mean/std and store it in the instance of StandardScaler
#.  2) fit_transform: Fitting on the data first and then perform z-scoring immediately.
#.  3) .inverse_transform() method: The method to use to turn the prediction output back to un-zscored real world value
#.     Ensure call .cpu().numpy() first on the prediction tensor before passing to inverse_transform() method.

feature_cols = data.keys().drop(['residual sugar'])
features_t = torch.tensor(data[feature_cols].values, dtype=torch.float32)
targets_t = torch.tensor(data['residual sugar'].values, dtype=torch.float32).reshape(-1, 1)

train_x_raw, test_x_raw, train_y_raw, test_y_raw = train_test_split(features_t, targets_t, test_size=0.2)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

to_tensor_float32 = lambda arr: torch.tensor(arr, dtype=torch.float32)

train_x_normal = to_tensor_float32( scaler_x.fit_transform(to_numpy(train_x_raw)) )
test_x_normal = to_tensor_float32( scaler_x.transform(to_numpy(test_x_raw)) )
train_y_normal = to_tensor_float32( scaler_y.fit_transform(to_numpy(train_y_raw)) )
test_y_normal = to_tensor_float32( scaler_y.transform(to_numpy(test_y_raw)) )

train_ds = TensorDataset(train_x_normal, train_y_normal)
test_ds = TensorDataset(test_x_normal, test_y_normal)
