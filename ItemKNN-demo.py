import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
import numpy as np
from sklearn.metrics import ndcg_score
import random

# 设置随机种子保证结果可复现
random.seed(42)
np.random.seed(42)


def load_movielens_1m_data():
    """
    加载 MovieLens-1M 数据集
    :return: 加载后的数据集和原始数据 DataFrame
    """
    file_path = '/Users/parker/Desktop/其它/ml-1m/ratings.dat'
    data = pd.read_csv(file_path, sep='::', header=None, engine='python',
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    data = data.sort_values(by='timestamp')
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    return dataset, data


def split_source_target(data):
    """
    划分源域数据、新用户子集和新物品子集
    :param data: 原始数据 DataFrame
    :return: 源域数据、新用户子集、新物品子集
    """
    source_data = data

    all_users = data['user_id'].unique()
    new_users = random.sample(list(all_users), int(len(all_users) * 0.1))
    new_user_subset = data[data['user_id'].isin(new_users)]

    item_interaction_counts = data['item_id'].value_counts()
    cold_start_items = item_interaction_counts[item_interaction_counts < 5].index
    new_item_subset = data[data['item_id'].isin(cold_start_items)]

    return source_data, new_user_subset, new_item_subset


def split_train_val_test(data):
    """
    按 8:1:1 比例划分训练集、验证集和测试集
    :param data: 输入数据 DataFrame
    :return: 训练集、验证集、测试集
    """
    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    reader = Reader(rating_scale=(1, 5))
    train_dataset = Dataset.load_from_df(train_data[['user_id', 'item_id', 'rating']], reader)
    val_dataset = Dataset.load_from_df(val_data[['user_id', 'item_id', 'rating']], reader)
    test_dataset = Dataset.load_from_df(test_data[['user_id', 'item_id', 'rating']], reader)

    return train_dataset, val_dataset, test_dataset


def train_itemknn_model(train_dataset):
    """
    训练 ItemKNN 模型
    :param train_dataset: 训练数据集
    :return: 训练好的模型
    """
    trainset = train_dataset.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    return algo


def calculate_hr(predictions, k_values=[5, 10]):
    """
    计算 HR@K 指标
    :param predictions: 模型预测结果
    :param k_values: K 的取值列表，默认为 [5, 10]
    :return: HR@K 的平均值
    """
    hr_scores = []
    for k in k_values:
        hit_count = 0
        total_count = 0
        user_predictions = {}
        for pred in predictions:
            user_id = pred.uid
            item_id = pred.iid
            true_rating = pred.r_ui
            if user_id not in user_predictions:
                user_predictions[user_id] = []
            user_predictions[user_id].append((item_id, true_rating, pred.est))

        for user_preds in user_predictions.values():
            sorted_preds = sorted(user_preds, key=lambda x: x[2], reverse=True)
            top_k_items = [item[0] for item in sorted_preds[:k]]
            relevant_items = [item[0] for item in user_preds if item[1] > 0]
            for item in relevant_items:
                if item in top_k_items:
                    hit_count += 1
                total_count += 1

        if total_count > 0:
            hr = hit_count / total_count
            hr_scores.append(hr)

    return np.mean(hr_scores) if hr_scores else 0


def calculate_ndcg(predictions, k_values=[5, 10]):
    """
    计算 NDCG@K 指标
    :param predictions: 模型预测结果
    :param k_values: K 的取值列表，默认为 [5, 10]
    :return: NDCG@K 的平均值
    """
    ndcg_scores = []
    for k in k_values:
        user_ndcg = {}
        for pred in predictions:
            user_id = pred.uid
            if user_id not in user_ndcg:
                user_ndcg[user_id] = ([], [])
            user_ndcg[user_id][0].append(pred.r_ui)
            user_ndcg[user_id][1].append(pred.est)

        user_ndcg_scores = []
        for true_ratings, pred_ratings in user_ndcg.values():
            if len(true_ratings) > 1:
                ndcg = ndcg_score([true_ratings], [pred_ratings], k=k)
                user_ndcg_scores.append(ndcg)

        if user_ndcg_scores:
            ndcg_scores.append(np.mean(user_ndcg_scores))

    return np.mean(ndcg_scores) if ndcg_scores else 0


def evaluate_model(algo, test_dataset):
    """
    评估模型，计算 RMSE、HR@K 和 NDCG@K
    :param algo: 训练好的模型
    :param test_dataset: 测试数据集
    :return: RMSE、HR@K、NDCG@K
    """
    testset = test_dataset.build_full_trainset().build_testset()
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    hr = calculate_hr(predictions)
    ndcg = calculate_ndcg(predictions)
    return rmse, hr, ndcg


def calculate_weighted_avg(source_metrics, new_user_metrics, new_item_metrics,
                           source_size, new_user_size, new_item_size):
    """
    计算整体加权平均值
    :param source_metrics: 源域指标列表 [RMSE, HR@K, NDCG@K]
    :param new_user_metrics: 新用户子集指标列表 [RMSE, HR@K, NDCG@K]
    :param new_item_metrics: 新物品子集指标列表 [RMSE, HR@K, NDCG@K]
    :param source_size: 源域数据量
    :param new_user_size: 新用户子集数据量
    :param new_item_size: 新物品子集数据量
    :return: 加权后的 RMSE、HR@K、NDCG@K
    """
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
    dataset, data = load_movielens_1m_data()
    source_data, new_user_subset, new_item_subset = split_source_target(data)

    source_train, source_val, source_test = split_train_val_test(source_data)
    new_user_train, new_user_val, new_user_test = split_train_val_test(new_user_subset)
    new_item_train, new_item_val, new_item_test = split_train_val_test(new_item_subset)

    source_algo = train_itemknn_model(source_train)
    source_rmse, source_hr, source_ndcg = evaluate_model(source_algo, source_test)
    print("Source Domain:")
    print(f'RMSE: {source_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {source_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {source_ndcg}')

    new_user_algo = train_itemknn_model(new_user_train)
    new_user_rmse, new_user_hr, new_user_ndcg = evaluate_model(new_user_algo, new_user_test)
    print("\nNew User Subset:")
    print(f'RMSE: {new_user_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {new_user_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {new_user_ndcg}')

    new_item_algo = train_itemknn_model(new_item_train)
    new_item_rmse, new_item_hr, new_item_ndcg = evaluate_model(new_item_algo, new_item_test)
    print("\nNew Item Subset:")
    print(f'RMSE: {new_item_rmse}')
    print(f'HR@K (avg of K=5 and K=10): {new_item_hr}')
    print(f'NDCG@K (avg of K=5 and K=10): {new_item_ndcg}')

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
