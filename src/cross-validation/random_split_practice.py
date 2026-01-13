import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

sample_size = 100

# dummy data
X = torch.randn(sample_size, 4)
y = torch.randint(0, 2, (sample_size,))
dataset = TensorDataset(X, y)

# split sizes
train_size = int(sample_size * 0.7)
val_size = int(sample_size * 0.15)
test_size = sample_size - train_size - val_size

train_data, val_data, test_data = random_split(
  dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
val_loader = DataLoader(val_data, batch_size=6, shuffle=False)
test_loader = DataLoader(test_data, batch_size=6, shuffle=False)
