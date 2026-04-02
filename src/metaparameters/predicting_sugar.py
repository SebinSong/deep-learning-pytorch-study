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

F = nn.functional

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Sanitize the raw-data from training

# 1. drop 'quality' column - Don't need it in predicting the residual sugar
data.drop(columns=['quality'], inplace=True)

# 2. drop some outliers
data = data[data['total sulfur dioxide'] < 200]

feature_cols = data.keys().drop(['residual sugar'])
D_in = len(feature_cols)
D_out = 1

# 3. Normalize the data
# - using 'StandardScaler' here to perform z-scoring on the data and then
#   reverse transform the predicted data later (it remembers the mean/std info of the fitted data)
#
# some terminologies:
#.  1) fitting -> Computing mean/std and store it in the instance of StandardScaler
#.  2) fit_transform: Fitting on the data first and then perform z-scoring immediately.
#.  3) .inverse_transform() method: The method to use to turn the prediction output back to un-zscored real world value
#.     Ensure call .cpu().numpy() first on the prediction tensor before passing to inverse_transform() method.

def split_data(train_prop=0.8, batch_size=32):
  global feature_cols

  features_t = torch.tensor(data[feature_cols].values, dtype=torch.float32)
  targets_t = torch.tensor(data['residual sugar'].values, dtype=torch.float32).reshape(-1, 1)

  train_x_raw, test_x_raw, train_y_raw, test_y_raw = train_test_split(features_t, targets_t, test_size=(1 - train_prop))

  scaler_x = StandardScaler()
  scaler_y = StandardScaler()

  to_tensor_float32 = lambda arr: torch.tensor(arr, dtype=torch.float32)

  train_x_normal = to_tensor_float32( scaler_x.fit_transform(to_numpy(train_x_raw)) )
  test_x_normal = to_tensor_float32( scaler_x.transform(to_numpy(test_x_raw)) )
  train_y_normal = to_tensor_float32( scaler_y.fit_transform(to_numpy(train_y_raw)) )
  test_y_normal = to_tensor_float32( scaler_y.transform(to_numpy(test_y_raw)) )

  train_ds = TensorDataset(train_x_normal, train_y_normal)
  test_ds = TensorDataset(test_x_normal, test_y_normal)
  
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = DataLoader(test_ds, batch_size=len(test_ds))

  return (
    train_loader,
    test_loader,
    scaler_x,
    scaler_y
  )

# define the predictor model
class ANNSugarPredictor(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.ModuleDict({
      'input': nn.Linear(D_in, 32),
      'hidden0': nn.Linear(32, 32),
      'bnorm0': nn.BatchNorm1d(32),
      'hidden1': nn.Linear(32, 16),
      'bnorm1': nn.BatchNorm1d(16),
      'output': nn.Linear(16, D_out),
    })

  def forward(self, x):
    x = F.relu( self.layers['input'](x) )

    # Linear -> Batch Norm -> Activation
    x = self.layers['hidden0'](x)
    x = F.relu( self.layers['bnorm0'](x) )

    x = self.layers['hidden1'](x)
    x = F.relu( self.layers['bnorm1'](x) )

    return self.layers['output'](x)

def train_model(train_ldr, test_ldr, learning_rate=0.01, num_epochs=800):
  model = ANNSugarPredictor()
  loss_func = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  all_train_losses = np.zeros(num_epochs)
  all_test_losses = np.zeros(num_epochs)

  for epoch_i in range(num_epochs):
    model.train()

    all_batch_losses = []
    for batch_x, batch_y in train_ldr:
      y_hat = model(batch_x)
      loss = loss_func(y_hat, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_batch_losses.append( loss.item() )

    all_train_losses[epoch_i] = np.mean(all_batch_losses)

    # compute the test loss
    model.eval()
    test_x, test_y = next(iter(test_ldr))

    with torch.no_grad():
      predictions = model(test_x)
      test_loss = loss_func(predictions, test_y)
    all_test_losses[epoch_i] = test_loss.item()

  return all_train_losses, all_test_losses, model
  
num_epochs = 800
train_loader, test_loader, scaler_x, scaler_y = split_data(batch_size=64)
train_losses, test_losses, model = train_model(train_loader, test_loader, num_epochs=num_epochs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.plot(train_losses, label='Train loss')
ax1.plot(test_losses, label='Test loss')
ax1.set_title('Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

train_ds = train_loader.dataset
test_ds = test_loader.dataset

train_x, train_labels = train_ds.tensors
test_x, test_labels = test_ds.tensors

# final predictions
with torch.no_grad():
  preds_train = model(train_x)
  preds_test = model(test_x)

ax2.plot(preds_train.detach(), train_labels, 'ro')
ax2.plot(preds_test.detach(), test_labels, 'b^')
ax2.set_xlabel('Model predicted sugar')
ax2.set_ylabel('True sugar')
ax2.set_title('Model preds vs Observations')

# Detach from graph -> Move to CPU (if on GPU) -> Convert to NumPy
detach_tensor = lambda t: t.detach().cpu().numpy()

corr_train = np.corrcoef(detach_tensor(preds_train).T, detach_tensor(train_labels).T)[1,0]
corr_test = np.corrcoef(detach_tensor(preds_test).T, detach_tensor(test_labels).T)[1,0]
ax2.legend([f'Train r={corr_train:.3f}', f'Test r={corr_test:.3f}'])

print('preds_test: ', preds_test)
print('test_labels: ', test_labels)

plt.show()
