import torch
import torch.nn as nn

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
norm_layer = nn.LayerNorm(3)
input_features = torch.tensor([
  [1, 2, 3],
  [4, 5, 6]
], dtype=torch.float32)
normalized_features = norm_layer(input_features)
print(normalized_features)
