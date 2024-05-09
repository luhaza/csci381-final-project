import torch
import torch_geometric

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Embedding, Linear, init
from representations import build_interaction_matrix_from_edges

class LightGCN(MessagePassing):

    def __init__(self, num_src, num_dest, dropout_rate=0.1, embedding_dim=64, layers=3):
        super().__init__()
        self.num_src = num_src
        self.num_dest = num_dest
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.layers = layers

        self.src_emb = Embedding(num_embeddings=self.num_src, 
                               embedding_dim=self.embedding_dim)
        self.dest_emb = Embedding(num_embeddings=self.num_dest, 
                               embedding_dim=self.embedding_dim)
        
        init.normal_(self.src_emb.weight, std=0.1)
        init.normal_(self.dest_emb.weight, std=0.1)

        self.out = Linear(embedding_dim + embedding_dim, 1)

    def forward(self, edges, values):
        # edges : edge indices (2xN matrix)
        # values : value for each link (1xN matrix)

        # compute symmetrical normalization A~
        norm = gcn_norm(edges, add_self_loops=False)

        e_0 = torch.cat([self.src_emb.weight, self.dest_emb.weight])
        e = [e_0]
        e_k = e_0

        for i in range(self.layers):
            e_k = self.propagate(edge_index=norm[0], x=e_k, norm=norm[1])
            e.append(e_k)

        e = torch.stack(e, dim=1)
        final = torch.mean(e, dim=1)
        user_emb, item_emb = torch.split(final, [self.num_src, self.num_dest])
        indices = 

        raise NotImplementedError("TODO")

    def message(self, x, norm):
        return norm.view(-1, 1) * x
