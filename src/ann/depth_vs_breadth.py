import torch

nn = torch.nn

widenet = nn.Sequential(
  nn.Linear(2, 4), # hidden layer
  nn.Linear(4, 3) # output layer
)

deepnet = nn.Sequential(
  nn.Linear(2, 2), # hidden layer
  nn.Linear(2, 2), # hidden layer
  nn.Linear(2, 3) # output layer
)

# num_node_in_widenet = 0
# for p in widenet.named_parameters():
#   if 'bias' in p[0]:
#     num_node_in_widenet += p[1].numel()

# num_node_in_deepent = 0
# for pName, pVector in deepnet.named_parameters():
#   if 'bias' in pName:
#     num_node_in_deepent += pVector.numel()

# print(f'num of nodes wide: {num_node_in_widenet}')
# print(f'num of nodes deep: {num_node_in_deepent}')

n_params_wide = sum([ p.numel() for p in widenet.parameters() if p.requires_grad ])
n_params_deep = sum([ p.numel() for p in deepnet.parameters() if p.requires_grad ])

print('n_params_wide: ', n_params_wide)
print('n_params_deep: ', n_params_deep)
