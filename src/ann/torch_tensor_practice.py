import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

## 1. nnLinear()

# # Number of data points
# N = 10

# # Define dimensions
# D_in = 1
# D_out = 1

# # Create the Linear layer
# linear_layer = nn.Linear(in_features=1, out_features=D_out)

# # We can look inside and see the parameters it created for us
# print(f'Layer\'s Weight (W): {linear_layer.weight}')
# print(f'Layer\'s Bias (b): {linear_layer.bias}')

# # dataset
# X = torch.randn(N, D_in)

# y_hat_nn = linear_layer(X)
# print(f'Output of nn.Linear (first 3 rows): {y_hat_nn[:3]}')

## 2. nn.ReLU() and nn.GELU()

# t1 = torch.linspace(-3, 3, 10)
# relu = nn.ReLU()
# gelu = nn.GELU()
# activated_t1 = relu(t1)
# activated_t1_2 = gelu(t1)

# print(t1)
# print(activated_t1)
# print(activated_t1_2)

## 3. nn.Softmax()
# softmax = nn.Softmax(dim=-1)

# logits = torch.tensor([
#   [1.0, 3.0, -0.5, 1.5],
#   [-1.0, 2.0, 1.0, 0.0]
# ], dtype=torch.float32)

# probs = softmax(logits)

# print(f'Logits: {logits}')
# print(f'probabilities: {probs}')
# print(f'sum of probs: {probs[0].sum()}')

# predicted_class = torch.argmax(probs, dim=1)
# print(f'predicted class: {predicted_class}')

## 4. nn.Embedding()

# vocab_size = 10 # Our language has 10 unique words
# embedding_dim = 3 # We'll represent each word with a 3D vector

# embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# # Input: A sentence where each word is an ID. eg. 1, 5, 0, 8
# input_ids = torch.tensor([1, 5, 0, 8])

# word_vectors = embedding_layer(input_ids)
# print('word_vectors for this sentence: ', word_vectors)

# 5. nn.LayerNorm()
# norm_layer = nn.LayerNorm(3)
# input_features = torch.tensor([
#   [1, 2, 3],
#   [4, 5, 6]
# ], dtype=torch.float32)
# normalized_features = norm_layer(input_features)
# print(normalized_features)

# arr = np.array([[10, 15, 20, 25, 30]])
# result = np.where(arr > 20)
# print(arr > 20)
# print(result)

# t1 = torch.tensor([
#   [2, 4, 6, 8],
#   [12, 14, 16, 18],
#   [21, 22, 33, 44]
# ])

# clusterSize = 5
# blur = 1.5

# center = {
#   'A': (2, 3),
#   'B': (7, 3)
# }

# a = np.array([
#   center['A'][0] + np.random.randn(clusterSize) * blur,
#   center['A'][1] + np.random.randn(clusterSize) * blur
# ], dtype=np.float32).T
# b = np.array([
#   center['B'][0] + np.random.randn(clusterSize) * blur,
#   center['B'][1] + np.random.randn(clusterSize) * blur
# ]).T

# at = torch.from_numpy(a)
# bt = torch.from_numpy(b)
# stacked = torch.vstack((at, bt))
# labels = torch.vstack((
#   torch.zeros(at.shape[0], 1),
#   torch.ones(bt.shape[0], 1)
# ))

# print(stacked)
# print(labels)

# t1 = torch.rand(10, 1)
# t2 = torch.rand(10, 1) > 0.5
# floated_t2 = t2.float()

# print(t1)
# print(floated_t2.mean())

# t_labels = torch.tensor([
#   0, 1, 1, 0, 1, 1, 0
# ], dtype=torch.float32)
# t_prediction = torch.tensor([
#   -1.23, 5.12, -3.5, -2.1, 10.5, -1.3, -42.11
# ], dtype=torch.float32)
# t_transformed_prediction = torch.where(t_prediction > 0, 1, 0).type(torch.float32)

# compared = (t_transformed_prediction == t_labels).float().mean() * 100
# print(f'{compared:.2f}%')

# t1 = torch.full((10,), 1.5).type(torch.float)
# print(t1.numel())

# num_rows = 5
# learning_rates = np.linspace(0.001, 0.1, num_rows, dtype=np.float32)
# accuracies = np.zeros((num_rows, 2))
# for i, lr in enumerate(learning_rates):
#   accuracies[i, :] = [i, lr]

# print(accuracies)

N = 50

D_in = 1
D_out = 1

X = torch.randn(N, D_in)

class LinearRegressionModel(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.linear_layer = nn.Linear(in_features, 16)
    self.activation_layer = nn.ReLU()
    self.output_layer = nn.Linear(16, out_features)

  def forward(self, X):
    # In the forward pass, we connect the layers
    t1 = self.linear_layer(X)
    t1_0 = self.activation_layer(t1)
    return self.output_layer(t1_0)

model = LinearRegressionModel(D_in, D_out)

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 1.5

learning_rate, epochs = 0.01, 350

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

tracked_losses = np.zeros(epochs, dtype=float)

for epoch in range(epochs):
  y_hat = model(X)

  loss = loss_fn(y_hat, y_true)

  tracked_losses[epoch] = loss.item()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# final prediction
predictions = model(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(X[:, 0], y_true[:, 0], 'bo', label='True data')
ax[0].plot(X[:, 0], predictions.detach()[:, 0], 'ro', label='Prediction data')
ax[0].set_xlabel('input')
ax[0].set_ylabel('output')
ax[0].legend()
ax[0].set_title('Real vs Final prediction')

ax[1].plot(np.arange(epochs), tracked_losses, 'bo-', markerfacecolor='w')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].set_title('Loss change')

plt.show()