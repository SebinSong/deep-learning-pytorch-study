import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

curr_dir = Path(__file__).parent
resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()

# helpers
unique_val = lambda nd_arr: torch.unique(nd_arr).sort()[0]

# path to dataset csv
data_fpath = resolve_path('../data/mnist_train_small.csv')

def read_data(data_path: str, no_header=False) -> any:
  if data_path.is_file():
    return pd.read_csv(data_path, header=(None if no_header else 0))
  
  return None

# load MNIST dataset and split into data/labels
df = read_data(data_fpath, no_header=True)
data = torch.tensor(df.values[:, 1:], dtype=torch.float32)
labels = torch.tensor(df.values[:, 0], dtype=torch.long)

sample_size = len(data)
unique_labels = unique_val(labels)

draw_label_histogram = False

if draw_label_histogram:
  sns.set_theme()
  sns.histplot(labels, binwidth=1)
  plt.xticks(ticks=(unique_labels + 0.5), labels=list(range(10)))
  plt.show()

def display_few_images():
  _, axs = plt.subplots(3, 4, figsize=(10, 6))
  flattened_axs = axs.flatten() # This is 1-dimensional ndarray
  random_img_idxs = np.random.randint(0, sample_size, 3*4)

  for ax_i, ax in enumerate(flattened_axs):
    img_idx = random_img_idxs[ax_i]

    img_mat = data[img_idx].reshape((28, -1)) # 28 x 28 matrix of the image
    img_label = labels[img_idx]

    ax.imshow(img_mat, cmap='gray')
    ax.set_title(f'Label: {img_label}')

  plt.suptitle('How humans see the data', fontsize=20)
  plt.tight_layout()
  plt.show()

def display_computerized_images():
  _, axs = plt.subplots(3, 4, figsize=(10, 6))

  flattened_axs = axs.flatten()
  random_img_idxs = np.random.randint(0, sample_size, 3*4)

  for ax_i, ax in enumerate(flattened_axs):
    img_idx = random_img_idxs[ax_i]

    img_data = data[img_idx]
    img_label = labels[img_idx]

    ax.plot(img_data, 'ko')
    ax.set_title(f'Label: {img_label}')
  
  plt.suptitle('How the FFN model sees the data')
  plt.tight_layout()
  plt.show()

display_computerized_images()
