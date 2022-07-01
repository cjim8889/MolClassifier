import torch
from torch import nn
from egnn_pytorch import EGNN


class Classfier(nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=64, gnn_size=3) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim)

        self.net = nn.ModuleList(
            [EGNN(embedding_dim, m_dim=hidden_dim, num_nearest_neighbors=5, norm_coors=True, soft_edges=True, coor_weights_clamp_value=2.) for _ in range(gnn_size)]
        )


        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(1),
            # nn.Sigmoid()
        )
    
    def forward(self, atom_types, pos, mask=None):
        feats = self.embedding(atom_types) * mask.unsqueeze(2)
        coors = pos * mask.unsqueeze(2)

        for net in self.net:
            feats, coors = net(feats, coors, mask=mask)

        feats = torch.sum(feats * mask.unsqueeze(2), dim=1) / torch.sum(mask, dim=1, keepdim=True)
        output = self.mlp(feats)


        return output