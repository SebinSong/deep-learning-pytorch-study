import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. Create dummy data (e.g., 1000 samples, 10 features each)
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,)) # Binary labels
dataset = TensorDataset(X, y)

# 2. Define split sizes
total_size = len(dataset)
train_size = int(0.7 * total_size) # 70% for Training
val_size = int(0.15 * total_size) # 15% for Validation
test_size = total_size - train_size - val_size # Remaining 15% for Test

# 3. Randomly split the dataset
# This is crucial: random_split ensures the data is shuffled before splitting
train_data, val_data, test_data = random_split(
  dataset, [train_size, val_size, test_size]
)

print(f'Train size: {len(train_data)}')
print(f'Val size: {len(val_data)}')
print(f'Test size: {len(test_data)}')

# 4. Create DataLoaders for each set
# Note: We usually shuffle the Training set, but NOT the Val/Test sets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 5. Example training loop structure
def train_one_epoch(model, loss_func, optimizer):
  # --- TRAINING PHASE ---
  # model.train()
  for batch_X, batch_y in train_loader:
    optimizer.zero_grad()
    predictions = model(batch_X)
    loss = loss_func(predictions, batch_y)
    loss.backward()
    optimizer.step()

  # --- VALIDATION PHASE ---
  # model.eval() with torch.no_grad():
  val_loss = 0.0
  # for batch_X, batch_y in val_loader:
  #   predictions = model(batch_X)
  #   loss = loss_func(predictions, batch_y)
  # ...
