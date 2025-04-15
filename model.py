import torch
import torch.nn as nn


class FeatureTransfer(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        return self.user_embed(user_ids), self.item_embed(item_ids)


class MTRec(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()
        self.feature_transfer = FeatureTransfer(num_users, num_items, embed_dim)
        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=2
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_ids, item_ids, seq_items):
        # 用户和物品嵌入
        user_emb, item_emb = self.feature_transfer(user_ids, item_ids)

        # 序列编码
        seq_emb = self.feature_transfer.item_embed(seq_items).permute(1, 0, 2)
        seq_output = self.sequence_encoder(seq_emb)[-1]

        # 拼接特征
        combined = torch.cat([user_emb, seq_output, item_emb], dim=-1)
        return self.predictor(combined).squeeze()