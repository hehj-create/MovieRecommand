import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(data_path, item_path):
    ratings = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(item_path, sep='|', encoding='latin-1',
                         names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

    # 编码用户和物品ID
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    ratings['user_id'] = user_enc.fit_transform(ratings.user_id)
    ratings['item_id'] = item_enc.fit_transform(ratings.item_id)

    # 生成用户行为序列（最后10次交互）
    user_sequences = ratings.sort_values(['user_id', 'timestamp']).groupby('user_id')['item_id'].apply(list)
    user_sequences = user_sequences.apply(lambda x: x[-10:])  # 保留最近10次行为

    # 划分训练测试集
    train_data, test_data = train_test_split(ratings, test_size=0.2, stratify=ratings['user_id'])

    return {
        'train_data': train_data,
        'test_data': test_data,
        'user_sequences': user_sequences,
        'user_enc': user_enc,
        'item_enc': item_enc,
        'movies': movies
    }