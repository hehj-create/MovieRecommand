import torch
from torch.utils.data import DataLoader, Dataset
from model import MTRec
from utils import load_data
import numpy as np


class MovieLensDataset(Dataset):
    def __init__(self, data, user_sequences):
        self.users = data['user_id'].values
        self.items = data['item_id'].values
        self.ratings = data['rating'].values
        self.user_sequences = user_sequences

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user][:-1]  # 历史序列（排除最后一个）
        return {
            'user': torch.LongTensor([user]),
            'item': torch.LongTensor([self.items[idx]]),
            'seq': torch.LongTensor(seq[-10:] if len(seq) >= 10 else [0] * (10 - len(seq)) + seq),
            'rating': torch.FloatTensor([self.ratings[idx] / 5.0])
        }


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = load_data('data/ml-100k/u.data', 'data/ml-100k/u.item')

    # 数据集
    train_dataset = MovieLensDataset(data['train_data'], data['user_sequences'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 初始化模型
    model = MTRec(num_users=data['user_enc'].classes_.size,
                  num_items=data['item_enc'].classes_.size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            users = batch['user'].squeeze().to(device)
            items = batch['item'].squeeze().to(device)
            seqs = batch['seq'].to(device)
            ratings = batch['rating'].squeeze().to(device)

            preds = model(users, items, seqs)
            loss = torch.nn.MSELoss()(preds, ratings)

            # 对抗训练
            user_emb = model.feature_transfer.user_embed(users)
            fake_emb = torch.randn_like(user_emb)
            adv_loss = torch.nn.BCELoss()(
                model.feature_transfer.domain_discriminator(fake_emb),
                torch.ones(len(users), 1).to(device)
            )
            total_loss = loss + 0.1 * adv_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'mtrec_model.pth')


if __name__ == "__main__":
    train()