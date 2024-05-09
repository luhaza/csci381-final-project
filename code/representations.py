import torch

def to_sparse(A):
    return A.to_sparse_coo()

def to_dense(A):
    return A.to_dense()

def build_interaction_matrix_from_edges(E, r, c, V, use_value=True):
    R = torch.zeros((r, c))

    src = E[0]
    dest = E[-1]

    for e in range(src.shape[0]):
        i = src[e]
        j = dest[e]

        if use_value:
            R[i][j] = V[i]
        else: R[i][j] = 1

    return R

def convert_to_adj_matrix(A, r, c, V):
    # convert the interaction matrix to an adjacency matrix

    # r = A.shape[0]
    # c = A.shape[1]

    R = build_interaction_matrix_from_edges(A, r, c, V)

    r_zeros = torch.zeros((r,r))
    c_zeros = torch.zeros((c,c))

    right = torch.cat((R, c_zeros))
    left = torch.cat((r_zeros, R.t()))

    sparse = to_sparse(torch.cat((left, right), dim=1))

    return sparse.indices(), sparse.values()

def convert_to_dense_adj_matrix(A, r, c, V, use_value=True):
    R = build_interaction_matrix_from_edges(A, r, c, V, use_value=use_value)

    r_zeros = torch.zeros((r,r))
    c_zeros = torch.zeros((c,c))

    right = torch.cat((R, c_zeros))
    left = torch.cat((r_zeros, R.t()))

    return torch.cat((left, right), dim=1)

def extract_interaction_matrix(A, r, c, V,):
    # extract the original interaction matrix from the adjacency matrix (edge index)


    return A[:r, -c:] 