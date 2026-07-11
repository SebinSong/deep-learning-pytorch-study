import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

F = nn.functional

curr_dir = Path(__file__).parent
data_fpath = (curr_dir / '../data/mnist_train_small.csv').resolve()

data_df = pd.read_csv(data_fpath, sep=',', header=None)
data_t = torch.tensor( data_df.values[:, 1:], dtype=torch.float32 )
labels_t = torch.tensor( data_df.values[:, 0], dtype=torch.long )

# normalize data to the range of 0-1
data_t /= data_t.max()

# some global variables
num_epochs = 10

def occlude_single_img(img_t, thickness=3):
  cloned = img_t.clone().detach().reshape(28, -1)
  is_vertical = random.random() > 0.5
  start_idx = 4 + random.randint(0, 15)
  if is_vertical:
    cloned[:, start_idx:start_idx+thickness] = 1
  else:
    cloned[start_idx:start_idx+thickness, :] = 1
  
  return cloned.flatten()

def draw_img(img_t):
  reshaped = img_t.clone().detach().reshape(28, -1)

  plt.figure(figsize=(5, 5))
  plt.imshow(reshaped, cmap='gray')
  plt.show()

def occlude_and_draw(img_t, paint=False):
  occluded = occlude_single_img(img_t)
  if paint:
    occluded = occluded.reshape(28, -1)
    img_t = img_t.reshape(28, -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img_t, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')

    ax2.imshow(occluded, cmap='gray')
    ax2.set_title('Occluded')
    ax2.axis('off')

    plt.show()

  else:
    return occluded

class AE_MNIST(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential(
      nn.Linear(28 * 28, 128), # mat btw input and enc
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 28 * 28),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x)

def create_model(lr=0.0001):
  m = AE_MNIST()
  l = nn.MSELoss()
  o = torch.optim.Adam(m.parameters(), lr=lr)

  return m, l, o

def generate_batch_idxs(batch_size=32):
  idx_set = []
  sample_size = len(data_t)
  rand_idxs = torch.randperm(sample_size)
  num_set = int(sample_size / batch_size) # discarding the remnant has the same effect of drop_last+True

  for set_i in range(num_set):
    curr_i = set_i * batch_size
    idx_set.append(rand_idxs[curr_i:curr_i+batch_size])

  return idx_set

def train_model():
  model, loss_func, optimizer = create_model()
  batch_idxs = generate_batch_idxs()

  all_losses = []

  for epoch_i in range(num_epochs):
    model.train()
    print(f'Iterating the epoch #{epoch_i + 1}')

    for idx_mask in batch_idxs:
      batch = data_t[idx_mask]
      y_hat = model(batch)

      loss = loss_func(y_hat, batch)

      # back-prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      all_losses.append( loss.item() )

  print('Train done!')
  return all_losses, model

def occlude_imgs(num_imgs=30):
  idxs = np.random.choice(len(data_t), size=num_imgs)
  occluded_imgs = []

  for idx in idxs:
    occluded = occlude_and_draw(data_t[idx])
    occluded_imgs.append(occluded)

  stacked = torch.vstack(occluded_imgs)
  return stacked, idxs

losses, trained_model = train_model()
final_loss = np.mean( losses[-3:] )

occluded_imgs, occ_idxs = occlude_imgs(12)
deoccluded_imgs = trained_model(occluded_imgs)
coef_set = []

for img_i, de_img in enumerate(deoccluded_imgs):
  original_img = data_t[occ_idxs[img_i]].clone().detach().numpy()
  de_img = de_img.clone().detach().numpy()

  non_zero_idx = original_img > 0
  non_zero_original_img = original_img[non_zero_idx]
  non_zero_de_img = de_img[non_zero_idx]
  print(f'non_zero_len={len(non_zero_original_img)}')
  corr_coef = np.corrcoef(non_zero_original_img, non_zero_de_img)[0, 1]
  coef_set.append(corr_coef)
