import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/heart_disease_uci.csv').resolve()

df = pd.read_csv(data_fpath, header=0, sep=',')

# some global variables
D_in = 0
D_out = 1 # cuz it's binary classification
learning_rate = 0.001
num_epochs = 50

# some helper functions
def to_numpy (t):
  if t.requires_grad:
    # detach(): Stops tracking gradients
    return t.detach().cpu().numpy()
  return t.cpu().numpy()

def to_tensor_float32 (ndarr):
  # convert numpy arrary to a float32 tensor
  return torch.tensor(ndarr, dtype=torch.float32)

def draw_box_plot(df):
  plt.figure(figsize=(15, 8))
  sns.boxplot(data=df)
  plt.xticks(rotation=20)
  plt.show()


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

  # split data into features/labels and train/test datasets
  feature_cols = df.columns.drop('num')

  features_t = torch.tensor( df[feature_cols].values, dtype=torch.float32 )
  labels_t = torch.tensor( df['num'].values > 0, dtype=torch.float32 ).reshape(-1, 1)

  # TODO:
  # normalize the data by z-scoring them. (!IMPORTANT! - Do not scale 'num' column which will be used as targets)
  # split the data and labels and generate train/test data loaders

  return features_t, labels_t

def split_data(train_prop=.8, batch_size=16):
  global D_in

  features, labels = sanitize_data()
  D_in = features.shape[1]

  train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=(1 - train_prop))

  feature_scaler = StandardScaler()
  feature_scaler.fit(to_numpy(train_features)) # fitting -> calculating the mean/std of the given data

  perform_zscore = lambda t: to_tensor_float32( feature_scaler.transform(to_numpy(t)) )

  # INSIGHT: Why not scale before splitting? (The Leakage Trap)
  # If you scale the whole dataset before splitting,
  # the "Mean" of your training data will be influenced by the values in your test data.
  #
  # E.g.) The Insight: Your model shouldn't "know" the average cholesterol level of the patients it's going to be tested on later.
  #       If it does, information from the "future" (the test set) has leaked into the "past" (the training process),
  #       giving you over-optimistic accuracy scores that will fail in the real world.
  train_features_norm = perform_zscore(train_features)
  test_features_norm = perform_zscore(test_features)

  train_dset = TensorDataset(train_features_norm, train_labels)
  test_dset = TensorDataset(test_features_norm, test_labels)

  train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dset, batch_size=len(test_dset))

  return train_loader, test_loader

class HeartDisease_ANN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 32),
      'fc0': nn.Linear(32, 8, bias=False),
      'bnorm0': nn.BatchNorm1d(8),
      'fc1': nn.Linear(8, 4, bias=False),
      'bnorm1': nn.BatchNorm1d(4),
      'output': nn.Linear(4, D_out)
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc0'](x)
    x = F.relu( self.layers['bnorm0'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['fc1'](x)
    x = F.relu( self.layers['bnorm1'](x) )

    return self.layers['output'](x)

def create_model():
  m = HeartDisease_ANN()
  l = nn.BCEWithLogitsLoss()
  o = torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=1e-3)

  return m, l, o

def train_model(train_ldr, test_ldr):
  model, loss_func, optimizer = create_model()

  test_x, test_y = next(iter(test_ldr))

  all_train_acc = np.zeros(num_epochs)
  all_test_acc = np.zeros(num_epochs)
  all_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_acc = []
    all_batch_losses = []
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)

      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )
      
      # compute batch_accuracy
      batch_acc = ((y_hat > 0).float() == batch_y).float().mean() * 100
      all_batch_acc.append( batch_acc.item() )
  
    all_train_acc[epoch_i] = np.mean( all_batch_acc )
    all_losses[epoch_i] = np.mean( all_batch_losses )

    # compute test_accuracy
    model.eval()
    with torch.no_grad():
      pred_test_labels = (model(test_x) > 0).float()
    
    test_acc = (pred_test_labels == test_y).float().mean() * 100
    all_test_acc[epoch_i] = test_acc.item()

    print(f'[Epoch {epoch_i}] {test_acc=}, loss={np.mean( all_losses )}')
  
  return all_train_acc, all_test_acc, all_losses

train_loader, test_loader = split_data()
train_acc, test_acc, losses = train_model(train_loader, test_loader)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(losses)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Losses')
ax1.set_title('Losses')

ax2.plot(train_acc, label='Train')
ax2.plot(test_acc, label='Test')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Train - Test accuracies')
ax2.legend()

plt.tight_layout()
plt.show()
