import torch
import numpy as np
from model import MTRec
from utils import load_data


def hr_ndcg(model, test_data, user_sequences, item_enc, k=10):
    model.eval()
    device = next(model.parameters()).device
    hits, ndcgs = [], []

    for user_id in test_data['user_id'].unique():
        # 获取测试物品
        pos_item = test_data[test_data['user_id'] == user_id]['item_id'].values[0]

        # 生成候选物品（排除训练数据中的物品）
        all_items = np.arange(item_enc.classes_.size)
        with torch.no_grad():
            seq = torch.LongTensor(user_sequences[user_id][-10:]).unsqueeze(0).to(device)
            user_tensor = torch.LongTensor([user_id]).to(device)
            item_tensor = torch.LongTensor(all_items).to(device)

            scores = model(user_tensor.expand(len(all_items)),
                           item_tensor,
                           seq.expand(len(all_items), -1))
            scores = scores.cpu().numpy()

        # 计算HR@10和NDCG@10
        topk = np.argsort(scores)[-k:]
        if pos_item in topk:
            hits.append(1)
            rank = np.where(topk == pos_item)[0][0]
            ndcgs.append(1 / np.log2(rank + 2))
        else:
            hits.append(0)
            ndcgs.append(0)

    return np.mean(hits), np.mean(ndcgs)


if __name__ == "__main__":
    # 加载数据和模型
    data = load_data('data/ml-100k/u.data', 'data/ml-100k/u.item')
    model = MTRec(num_users=data['user_enc'].classes_.size,
                  num_items=data['item_enc'].classes_.size)
    model.load_state_dict(torch.load('mtrec_model.pth', map_location='cpu'))
    model.to('mps' if torch.backends.mps.is_available() else 'cpu')

    # 计算指标
    hr, ndcg = hr_ndcg(model, data['test_data'], data['user_sequences'], data['item_enc'])
    print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")