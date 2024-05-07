import torch
from representations import convert_to_adj_matrix, extract_interaction_matrix

# verify that helper functions are implemented correctly

A = torch.tensor([[0, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0]])

adj = convert_to_adj_matrix(A)

print(adj)
print(extract_interaction_matrix(adj, A.shape[0], A.shape[1]))