import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import ndcg_score
import random

# 设置随机种子以保证实验可重现性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# 定义 MovieLens 数据集类
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


# 定义 DARec 模型
class DARec(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(DARec, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        combined = torch.cat([user_embed, item_embed], dim=1)
        output = self.fc(combined)
        return output.squeeze()


# 计算 HR@K
def calculate_hr(predictions, k_values=[5, 10]):
    hr_values = []
    for k in k_values:
        user_hits = {}
        user_total = {}
        for prediction in predictions:
            user_id = prediction[0]
            item_id = prediction[1]
            true_rating = prediction[2]
            est_rating = prediction[3]
            if user_id not in user_hits:
                user_hits[user_id] = 0
                user_total[user_id] = 0
            user_total[user_id] += 1
            top_k_predictions = sorted([p for p in predictions if p[0] == user_id], key=lambda x: x[3], reverse=True)[
                                :k]
            top_k_items = [p[1] for p in top_k_predictions]
            if item_id in top_k_items and true_rating > 0:
                user_hits[user_id] += 1
        hr = np.mean([user_hits[user] / user_total[user] for user in user_hits if user_total[user] > 0])
        hr_values.append(hr)
    return np.mean(hr_values)


# 计算 NDCG@K
def calculate_ndcg(predictions, k_values=[5, 10]):
    ndcg_values = []
    for k in k_values:
        user_ndcg = {}
        for prediction in predictions:
            user_id = prediction[0]
            if user_id not in user_ndcg:
                user_ndcg[user_id] = []
            user_ndcg[user_id].append((prediction[2], prediction[3]))
        ndcg_scores = []
        for user, ratings in user_ndcg.items():
            true_ratings = [r[0] for r in ratings]
            pred_ratings = [r[1] for r in ratings]
            if len(true_ratings) > 1:
                ndcg = ndcg_score([true_ratings], [pred_ratings], k=k)
                ndcg_scores.append(ndcg)
        if ndcg_scores:
            ndcg_values.append(np.mean(ndcg_scores))
    return np.mean(ndcg_values) if ndcg_values else 0


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            outputs = model(user_ids, item_ids)
            for user_id, item_id, rating, output in zip(user_ids, item_ids, ratings, outputs):
                predictions.append((user_id.item(), item_id.item(), rating.item(), output.item()))
    rmse = np.sqrt(np.mean([(p[2] - p[3]) ** 2 for p in predictions]))
    hr = calculate_hr(predictions)
    ndcg = calculate_ndcg(predictions)
    return rmse, hr, ndcg


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for user_ids, item_ids, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')


# 加载 MovieLens-1M 数据
def load_movielens_1m_data():
    file_path = '/Users/parker/Desktop/其它/ml-1m/ratings.dat'
    data = pd.read_csv(file_path, sep='::', header=None, engine='python',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    data = data.sort_values(by='timestamp')
    return data


# 重新编号函数
def reindex_data(data):
    user_id_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
    item_id_mapping = {item_id: idx for idx, item_id in enumerate(data['item_id'].unique())}
    data['user_id'] = data['user_id'].map(user_id_mapping)
    data['item_id'] = data['item_id'].map(item_id_mapping)
    return data


# 划分源域和目标域数据
def split_source_target(data):
    # 源域数据为完整数据
    source_data = data

    # 新用户子集
    all_users = data['user_id'].unique()
    new_users = random.sample(list(all_users), int(len(all_users) * 0.1))
    new_user_subset = data[data['user_id'].isin(new_users)]

    # 新物品子集
    item_interaction_counts = data['item_id'].value_counts()
    cold_start_items = item_interaction_counts[item_interaction_counts < 5].index
    new_item_subset = data[data['item_id'].isin(cold_start_items)]

    return source_data, new_user_subset, new_item_subset


# 按 8:1:1 划分训练、验证和测试集
def split_train_val_test(data):
    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


# 计算整体加权平均值
def calculate_weighted_avg(source_metrics, new_user_metrics, new_item_metrics,
                           source_size, new_user_size, new_item_size):
    total_size = source_size + new_user_size + new_item_size
    source_weight = source_size / total_size
    new_user_weight = new_user_size / total_size
    new_item_weight = new_item_size / total_size

    weighted_rmse = (source_weight * source_metrics[0] +
                     new_user_weight * new_user_metrics[0] +
                     new_item_weight * new_item_metrics[0])
    weighted_hr = (source_weight * source_metrics[1] +
                   new_user_weight * new_user_metrics[1] +
                   new_item_weight * new_item_metrics[1])
    weighted_ndcg = (source_weight * source_metrics[2] +
                     new_user_weight * new_user_metrics[2] +
                     new_item_weight * new_item_metrics[2])
    return weighted_rmse, weighted_hr, weighted_ndcg


if __name__ == "__main__":
    data = load_movielens_1m_data()
    data = reindex_data(data)

    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()
    embedding_dim = 32

    # 源域数据处理
    source_data, new_user_subset, new_item_subset = split_source_target(data)
    source_train_data, source_val_data, source_test_data = split_train_val_test(source_data)
    source_train_dataset = MovieLensDataset(source_train_data)
    source_test_dataset = MovieLensDataset(source_test_data)
    source_train_loader = DataLoader(source_train_dataset, batch_size=64, shuffle=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=64, shuffle=False)

    source_model = DARec(num_users, num_items, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(source_model.parameters(), lr=0.001)
    train_model(source_model, source_train_loader, criterion, optimizer, epochs=10)
    source_rmse, source_hr, source_ndcg = evaluate_model(source_model, source_test_loader)
    print("Source Domain:")
    print(f'RMSE: {source_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {source_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {source_ndcg}')

    # 新用户子集数据处理
    new_user_train_data, new_user_val_data, new_user_test_data = split_train_val_test(new_user_subset)
    new_user_train_dataset = MovieLensDataset(new_user_train_data)
    new_user_test_dataset = MovieLensDataset(new_user_test_data)
    new_user_train_loader = DataLoader(new_user_train_dataset, batch_size=64, shuffle=True)
    new_user_test_loader = DataLoader(new_user_test_dataset, batch_size=64, shuffle=False)

    new_user_model = DARec(num_users, num_items, embedding_dim)
    optimizer = optim.Adam(new_user_model.parameters(), lr=0.001)
    train_model(new_user_model, new_user_train_loader, criterion, optimizer, epochs=10)
    new_user_rmse, new_user_hr, new_user_ndcg = evaluate_model(new_user_model, new_user_test_loader)
    print("\nNew User Subset:")
    print(f'RMSE: {new_user_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {new_user_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {new_user_ndcg}')

    # 新物品子集数据处理
    new_item_train_data, new_item_val_data, new_item_test_data = split_train_val_test(new_item_subset)
    new_item_train_dataset = MovieLensDataset(new_item_train_data)
    new_item_test_dataset = MovieLensDataset(new_item_test_data)
    new_item_train_loader = DataLoader(new_item_train_dataset, batch_size=64, shuffle=True)
    new_item_test_loader = DataLoader(new_item_test_dataset, batch_size=64, shuffle=False)

    new_item_model = DARec(num_users, num_items, embedding_dim)
    optimizer = optim.Adam(new_item_model.parameters(), lr=0.001)
    train_model(new_item_model, new_item_train_loader, criterion, optimizer, epochs=10)
    new_item_rmse, new_item_hr, new_item_ndcg = evaluate_model(new_item_model, new_item_test_loader)
    print("\nNew Item Subset:")
    print(f'RMSE: {new_item_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {new_item_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {new_item_ndcg}')

    # 计算整体加权平均值
    source_size = len(source_data)
    new_user_size = len(new_user_subset)
    new_item_size = len(new_item_subset)
    weighted_rmse, weighted_hr, weighted_ndcg = calculate_weighted_avg(
        [source_rmse, source_hr, source_ndcg],
        [new_user_rmse, new_user_hr, new_user_ndcg],
        [new_item_rmse, new_item_hr, new_item_ndcg],
        source_size, new_user_size, new_item_size
    )
    print("\nOverall Weighted Averages:")
    print(f'Weighted RMSE: {weighted_rmse}')
    print(f'Weighted HR@K (avg of K=5 and K=10): {weighted_hr}')
    print(f'Weighted NDCG@K (avg of K=5 and K=10): {weighted_ndcg}')
