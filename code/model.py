import torch
import random
import torch_geometric

from tqdm import tqdm
import scipy.sparse as sp # was having very annoying issues with torch_sparse
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.models.lightgcn import BPRLoss
from torch.nn import Embedding, init, functional
from torch_geometric.utils import structured_negative_sampling
from representations import extract_interaction_matrix
from torch_sparse import matmul

# training constants
ITERATIONS = 10000
BATCH_SIZE = 1024
LR = 1e-3
ITERS_PER_EVAL = 100
ITERS_PER_LR_DECAY = 200
K = 20
LAMBDA = 1e-6

# extend MessagePassing from PyG for Message/Aggregation
class LightGCN(MessagePassing):

    def __init__(self, num_users, num_items, embedding_dim=64, K=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K

        # embed initial users & items
        self.users_emb = Embedding(num_embeddings=self.num_users, 
                               embedding_dim=self.embedding_dim)
        self.items_emb = Embedding(num_embeddings=self.num_items, 
                               embedding_dim=self.embedding_dim)
        
        # normal initialization
        init.normal_(self.users_emb.weight, std=0.1)
        init.normal_(self.items_emb.weight, std=0.1)
    
    # edge index : adjacency matrix sparse tensor 
    def forward(self, edge_index):
        normalized_edge_index = gcn_norm(edge_index=edge_index, 
                                   add_self_loops=False)
        
        emb0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = []
        embs.append(emb0)

        # for each layer, propagate
        for _ in range(self.K):
            # emb0 is the only trainable embedding
            embk = self.propagate(normalized_edge_index, x=emb0)
            embs.append(embk)

        embs = torch.stack(embs, dim=1)

        # mean at embedding k
        mean_at_k = torch.mean(embs, dim=1)

        # split back into users and items at k
        u_k, i_k = torch.split(mean_at_k, [self.num_users, self.num_items])

        # must return all this for BPR loss calculation
        return u_k, self.users_emb.weight, i_k, self.items_emb.weight

    def message(self, x):
        return x
    
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x)

    
def bpr_loss(u_k, u_0, pos_i_k, pos_i_0, neg_i_k, neg_i_0, l):
    # BPR (Bayesian Personalized Ranking) loss function, as used in the paper

    regularized_loss = (u_0.norm(2).pow(2)
                        + pos_i_0.norm(2).pow(2)
                        + neg_i_0.norm(2).pow(2)) + l
    
    pos = torch.sum(torch.mul(u_k, pos_i_k), dim=1)
    neg = torch.sum(torch.mul(u_k, neg_i_k), dim=1)

    return -torch.mean(functional.softplus(pos-neg)) + regularized_loss

def pos_neg_minibatch(size, edges):
    # Minibatch for getting pos/neg samples and training, as used in the paper

    sample = torch.stack(structured_negative_sampling(edges), dim=0)
    all_index = list(range(sample[0].shape[0]))
    selected_index = random.choices(all_index, k=size)
    minibatch = sample[:, selected_index]

    return minibatch[0], minibatch[1], minibatch[2]

def to_device(things, device):
    for thing in things:
        thing.to(device)

def train_model(model, device, optimizer, scheduler,
                train_set, train_set_sparse):
    # Training loop

    training_loss = []

    for iter in tqdm(range(ITERATIONS)):
        u_k, u_0, i_k, i_0 = model.forward(train_set_sparse)

        u, pos, neg = pos_neg_minibatch(BATCH_SIZE, train_set)
        things = [u, pos, neg]
        to_device(things, device)

        u_k = u_k[u]
        u_0 = u_0[u]

        pos_k = i_k[pos]
        pos_0 = i_0[pos]

        neg_k = i_k[neg]
        neg_0 = i_0[neg]

        loss = bpr_loss(u_k, u_0, pos_k, pos_0, neg_k, neg_0, LAMBDA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % ITERS_PER_EVAL == 0:
            training_loss.append(loss.item())

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()
