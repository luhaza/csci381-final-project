import torch
import torch_geometric

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Embedding, init

class LightGCN(MessagePassing):

    def __init__(self, u, i, embedding_dim=64, K=3):
        super().__init__()
        self.u = u
        self.i = i
        self.embedding_dim = embedding_dim
        self.K = K

        self.u_emb = Embedding(num_embeddings=self.u, 
                               embedding_dim=self.embedding_dim)
        self.i_emb = Embedding(num_embeddings=self.i, 
                               embedding_dim=self.embedding_dim)
        
        init.normal_(self.u_emb.weight, std=0.1)
        init.normal_(self.i_emb.weight, std=0.1)

    def forward(self, x):
        raise NotImplementedError("TODO")

    def message():
        raise NotImplementedError("TODO")
