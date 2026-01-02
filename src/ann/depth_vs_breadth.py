import torch

# nn = torch.nn

# widenet = nn.Sequential(
#   nn.Linear(2, 4), # hidden layer
#   nn.Linear(4, 3) # output layer
# )

# deepnet = nn.Sequential(
#   nn.Linear(2, 2), # hidden layer
#   nn.Linear(2, 2), # hidden layer
#   nn.Linear(2, 3) # output layer
# )

# def get_num_nodes(model):
#   return sum([ p_params.numel() for p_name, p_params in model.named_parameters() if 'bias' in p_name ])

# def get_num_params(model):
#   return sum([ p_tensor.numel() for p_tensor in model.parameters() if p_tensor.requires_grad ])

# print(f'widenet - N of nodes: {get_num_nodes(widenet)}, N of trainable params: {get_num_params(widenet)}')
# print(f'deepnet - N of nodes: {get_num_nodes(deepnet)}, N of trainable params: {get_num_params(deepnet)}')

tup_list = [('a', 1), ('b', 2), ('c', 3)]
dict_from_tup = dict(tup_list)
dict1 = { k:v for k,v in tup_list }
print(dict_from_tup)
print(dict1)