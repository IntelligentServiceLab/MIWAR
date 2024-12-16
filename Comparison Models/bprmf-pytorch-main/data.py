import numpy as np
import torch


class DataLoader(object):
    def __init__(self, path="data/"):
        self.path = path
        self.allPos = []
        self.testPos = []
        self.m_item = 0
        self.cnt = 0
        with open(path + "train.txt", "r") as f:
            for line in f.readlines():
                x = list(map(int, line.split(' ')))
                self.allPos.append([])
                for item_id in x[1:]:
                    self.m_item = max(self.m_item, item_id + 1)
                    self.cnt += 1
                    self.allPos[-1].append(item_id)
        self.n_user = len(self.allPos)

        with open(path + "test.txt", "r") as f:
            for line in f.readlines():
                x = list(map(int, line.split(' ')))
                self.testPos.append(x[1])

    def generate_data(self, batch_size):
        users = np.random.randint(0, self.n_user, self.cnt)

        for i in range(0, self.n_user, batch_size):
            pos_items, neg_items = [], []
            for user in users[i: i+batch_size]:
                pos_items.append(np.random.choice(self.allPos[user]))
                t = np.random.randint(self.m_item)
                while t in self.allPos[user]:
                    t = np.random.randint(self.m_item)
                neg_items.append(t)
            yield users[i: i+batch_size], pos_items, neg_items


    def evaluate_metrics(self, model, device, k):
        recall_list = []
        ndcg_list = []
        precision_list = []
        for user_id in range(self.n_user):
            # 构造用户和物品列表
            users = [user_id] * (1 + 200)  # 一个正样本 + 99 个负样本
            items = [self.testPos[user_id]]  # 正样本
            # print("users :",users)
            # print("items :",items)
            for _ in range(200):  # 采样 99 个负样本
                t = np.random.randint(self.m_item)
                while t in self.allPos[user_id] or t in items:  # 避免重复
                    t = np.random.randint(self.m_item)
                items.append(t)

            # 转换为 tensor 并移动到设备
            users = torch.Tensor(users).long().to(device)
            items = torch.Tensor(items).long().to(device)

            # 模型预测评分
            scores = model.forward(users, items).cpu().numpy()
            # print(scores.shape)
            # 根据评分对物品排序
            sorted_indices = np.argsort(scores)[::-1]  # 按分数从高到低排序
            recommended_scores = np.sort(scores)[::-1][:k]
            top_k = sorted_indices[:k]  # 取前 k 个索引
            ground_truth_scores = []
            # 计算 Precision@k 和 Recall@k
            hits = 0
            for idx in top_k:
                if items[idx] == self.testPos[user_id]:  # 命中正样本
                    hits += 1
                    ground_truth_scores.append(1)
                else :
                    ground_truth_scores.append(0)
            precision = hits / k  # 推荐列表总长度为 k
            recall = hits / 1  # 每个用户只有一个正样本
            precision_list.append(precision)
            recall_list.append(recall)
            # 计算 NDCG@k
            dcg = 0


            ndcg = ndcg_at_k(recommended_scores, ground_truth_scores, k)
            ndcg_list.append(ndcg)
        # 返回 Recall, Precision 和 NDCG 的平均值
        print(np.sum(recall_list),np.sum(precision_list),np.sum(ndcg_list),len(precision_list))
        return np.mean(recall_list), np.mean(precision_list), np.mean(ndcg_list)




def dcg_at_k(scores, k):
    """
    计算 DCG@k
    :param scores: 排序后的相关性分数
    :param k: 前 k 个位置
    :return: DCG 值
    """
    scores = np.asfarray(scores)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(predicted_scores, true_scores, k):
    """
    计算 NDCG@k
    :param predicted_scores: 模型预测的分数
    :param true_scores: 实际的相关性分数
    :param k: 评价的前 k 个位置
    :return: NDCG 值
    """
    # 按预测分数排序后的实际分数
    sorted_true_scores = [true for _, true in sorted(zip(predicted_scores, true_scores), reverse=True)]

    # 计算 DCG 和 IDCG
    dcg = dcg_at_k(sorted_true_scores, k)
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)

    # 避免除以 0 的情况
    return dcg / idcg if idcg > 0 else 0.0
