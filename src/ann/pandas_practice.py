from pathlib import Path
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np

# base_path = Path(__file__).parent
# resolve_path = lambda rel_path: str((base_path / rel_path).resolve().as_posix())

# df = pd.DataFrame(
#   {
#     'Name': [
#       "Braund, Mr. Owen Harris",
#       "Allen, Mr. William Henry",
#       "Bonnell, Miss. Elizabeth"
#     ],
#     'Age': [22, 35, 58],
#     'Sex': ['male', 'male', 'female']
#   }
# )

# print(df.values.dtype)

# series_ages = pd.Series([22, 35, 58], name='Age', dtype=pd.Int64Dtype)

# print(f'average Age: {df["Age"].mean():_^12.3f}')

# print(type(series_ages.describe()))

# iris = sns.load_dataset('iris')
# first_5 = iris.head(5)
# dtypes = first_5.dtypes

# titanic = pd.read_csv(resolve_path('../data_samples/titanic.csv'))
# sns.pairplot(titanic, hue='Survived')
# plt.show()

# setup loss and inputs
# let's say we have 3 classes [Cat, Dog Bird]
loss_func = nn.NLLLoss()

# Mock output from a model (logits from 2 samples)
logits = torch.randn(2, 3)

# Convert to Log probabilities (Crucial step!)
log_probs = nn.functional.log_softmax(logits, dim=1)

# Target labels sample 1 is Dig, Sample 2 is Cat
targets = torch.tensor([1, 0])

output = loss_func(log_probs, targets)

print(f'The NLLLoss is: {output.item()}')
