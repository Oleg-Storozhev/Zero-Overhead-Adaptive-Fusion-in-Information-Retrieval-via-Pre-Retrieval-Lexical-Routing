import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class HybridLTRDataset(Dataset):
    def __init__(self, triplets):
        """
        triplets: list of dict like:
        {global_triplets
            'features': [f1, f2, ... f15],
            'pos_dense': float, 'pos_sparse': float,
            'neg_dense': float, 'neg_sparse': float
        }
        """
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        return {
            'x_q': torch.tensor(item['features'], dtype=torch.float32),
            's_d_pos': torch.tensor(item['pos_dense'], dtype=torch.float32),
            's_s_pos': torch.tensor(item['pos_sparse'], dtype=torch.float32),
            's_d_neg': torch.tensor(item['neg_dense'], dtype=torch.float32),
            's_s_neg': torch.tensor(item['neg_sparse'], dtype=torch.float32),
        }


class AlphaRouterMLP(nn.Module):
    def __init__(self, input_dim=17):
        super(AlphaRouterMLP, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_q):
        return self.net(x_q).squeeze(-1)


class AlphaRouter(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64):
        super(AlphaRouter, self).__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        self.res_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_res1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_q):

        gates = self.feature_gate(x_q)
        gated_x = x_q * gates

        h = F.gelu(self.bn_in(self.input_proj(gated_x)))

        res = self.res_layer1(h)
        res = self.bn_res1(res)
        res = self.dropout(F.gelu(res))

        out = h + res

        return self.head(out).squeeze(-1)