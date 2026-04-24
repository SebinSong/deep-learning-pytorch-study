import torch
import pandas as pd
from save_load_share_model import MNIST_ANN
from pathlib import Path

curr_dir = Path(__file__).parent

resolve_path = lambda rel_path: (curr_dir / rel_path).resolve()
model_data_path = resolve_path('../saved_models/MNIST_classifier1.pt')
test_datapath = resolve_path('../data/mnist_train_small.csv')

# load model_state
loaded_state_dict = torch.load(model_data_path)

model1 = MNIST_ANN()
model1.load_state_dict(loaded_state_dict)

# split data and labels
df = pd.read_csv(test_datapath, header=None, sep=',')
test_data = torch.tensor( df.values[:, 1:], dtype=torch.float32 )
test_data /= torch.max(test_data)

test_labels = torch.tensor( df.values[:, 0], dtype=torch.long )

with torch.no_grad():
  test_pred_labels = torch.argmax( model1(test_data), dim=1 )

accuracy = (test_pred_labels == test_labels).float().mean() * 100
print(f'Test accuracy: {accuracy:.3f}')
