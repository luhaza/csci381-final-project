import torch

def to_sparse(A):
    return A.to_sparse_coo()

def to_dense(A):
    return A.to_dense()

def convert_to_adj_matrix(A):
    # convert the interaction matrix to an adjacency matrix

    r = A.shape[0]
    c = A.shape[1]

    r_zeros = torch.zeros((r,r))
    c_zeros = torch.zeros((c,c))

    right = torch.cat((A, c_zeros))
    left = torch.cat((r_zeros, A.t()))

    return torch.cat((left, right), dim=1)

def extract_interaction_matrix(A, r, c):
    # extract the original interaction matrix from the adjacency matrix

    return A[0:r, -c:] 