import pandas as pd
import numpy as np
import math
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn
from utils import generate_unique_list
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义 SANFM 神经网络模型
random.seed(8080)
class SANFM(nn.Module):
    def __init__(self, embed_dim, att_dim,droprate=0.5, i_num=1536, c_num=2):
        super(SANFM, self).__init__()
        # 模型参数
        self.i_num = i_num  # 输入特征的数量
        self.c_num = c_num  # 类别特征的数量
        self.embed_dim = embed_dim  # 嵌入维度
        self.att_dim = att_dim
        self.bi_inter_dim = embed_dim  # 双交互池化输出维度
        self.droprate = droprate  # dropout 概率
        self.criterion = nn.BCELoss(weight=None, reduction='mean')  # 二分类交叉熵损失函数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

        # Dense embedding 层
        self.dense_embed = nn.Linear((self.i_num + self.c_num), self.embed_dim)

        # 双交互池化权重矩阵
        self.pairwise_inter_v = nn.Parameter(torch.empty(self.embed_dim, self.bi_inter_dim))

        # 自注意力机制的参数
        self.query_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.key_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.value_matrix = nn.Parameter(torch.empty(self.embed_dim, self.att_dim))
        self.softmax = nn.Softmax(dim=-1)  # 用于自注意力的 softmax

        # MLP 部分的全连接层
        self.hidden_1 = nn.Linear(self.att_dim, self.att_dim, bias=True)
        self.hidden_2 = nn.Linear(self.att_dim, 1)
        # self.hidden_1 = nn.Linear(self.embed_dim, 1)  # 第一个隐藏层
        # Batch Normalization 层
        self.bn = nn.BatchNorm1d(self.att_dim, momentum=0.5)

        # 权重初始化
        self._init_weight_()


    def BiInteractionPooling(self, pairwise_inter):
        # 计算二阶交互特征
        inter_part1_sum = torch.sum(pairwise_inter, dim=1)
        inter_part1_sum_square = torch.square(inter_part1_sum)  # square_of_sum

        inter_part2 = pairwise_inter * pairwise_inter
        inter_part2_sum = torch.sum(inter_part2, dim=1)  # sum of square
        bi_inter_out = 0.5 * (inter_part1_sum_square - inter_part2_sum)
        return bi_inter_out

    def _init_weight_(self):
        """初始化模型参数"""
        # dense embedding
        nn.init.normal_(self.dense_embed.weight, std=0.1) # 0.06
        # pairwise interaction pooling
        nn.init.normal_(self.pairwise_inter_v, std=0.1)
        # deep layers
        nn.init.kaiming_normal_(self.hidden_1.weight)
        nn.init.kaiming_normal_(self.hidden_2.weight)
        # attention part
        nn.init.kaiming_normal_(self.query_matrix)
        nn.init.kaiming_normal_(self.key_matrix)
        nn.init.kaiming_normal_(self.value_matrix)

    def forward(self, batch_data):
        batch_data = batch_data.to(torch.float32).to(device)

        # Dense embedding 计算
        dense_embed = self.dense_embed(batch_data).to(device)

        # 双交互池化计算
        pairwise_inter = dense_embed.unsqueeze(1) * self.pairwise_inter_v  # 3D 张量
        pooling_out = self.BiInteractionPooling(pairwise_inter)  # 2D 张量

        # 自注意力机制
        X = pooling_out
        proj_query = torch.mm(X, self.query_matrix)  # Query 矩阵
        proj_key = torch.mm(X, self.key_matrix)  # Key 矩阵
        proj_value = torch.mm(X, self.value_matrix)  # Value 矩阵

        S = torch.mm(proj_query, proj_key.T)  # Q * K^T
        attention_map = self.softmax(S)  # 计算注意力权重

        # 计算加权的 Value
        value_weight = proj_value[:, None] * attention_map.T[:, :, None]
        value_weight_sum = value_weight.sum(dim=0)  # 加权和

        # MLP 部分
        mlp_hidden_1 = F.relu(self.bn(self.hidden_1(value_weight_sum)))

        mlp_hidden_2 = F.dropout(mlp_hidden_1, training=self.training, p=self.droprate)
        mlp_out = self.hidden_2(mlp_hidden_2)
        # mlp_hidden_1 = F.relu(self.bn(self.hidden_1(value_weight_sum)))  # 第一层
        # mlp_hidden_2 = F.relu(self.hidden_2(mlp_hidden_1))  # 第二层
        # mlp_hidden_3 = F.relu(self.hidden_3(mlp_hidden_2))  # 第二层
        # mlp_hidden_4 = F.relu(self.hidden_4(mlp_hidden_3))  # 第二层
        # mlp_hidden_5 = F.dropout(mlp_hidden_4, training=self.training, p=self.droprate)  # Dropout
        # mlp_out = self.hidden_1(value_weight_sum)  # 第三层（输出层）
        final_sig_out = self.sigmoid(mlp_out)
        final_sig_out_squeeze = final_sig_out.squeeze()
        return final_sig_out_squeeze

    def loss(self, batch_input, batch_label):
        pred = self.forward(batch_input)
        pred = pred.to(torch.float32)
        pred = pred.to(device)
        batch_label = batch_label.to(torch.float32).squeeze().to(device)
        # print("pred = ",pred,"batch_label = ",batch_label)
        Loss = self.criterion(pred, batch_label)  # 计算损失
        return Loss


# 训练函数
def SANFM_train(model, inputs,labels):
    model.train()
    loss = model.loss(inputs, labels).to(device)
    return loss

# 测试函数
def tst(model, test_loader):
    model.eval()
    criterion = nn.BCELoss(reduction='mean')
    LOSS = []
    AUC = []

    # 测试集的每个batch
    for test_input, test_label in test_loader:
        pred = model(test_input)
        pred = pred.to(torch.float32)
        test_label = test_label.squeeze().to(torch.float32)
        # print(pred)
        # 计算损失和AUC
        loss_value = criterion(pred, test_label)
        # print("pred = ",pred,"test_label = ",test_label,"shape of pred = ",pred.shape,"shape of test_label = ",test_label.shape)
        auc_value = roc_auc_score(test_label.detach().tolist(), pred.detach().tolist())

        LOSS.append(loss_value.detach())
        AUC.append(auc_value)

    # 计算平均损失和 AUC
    loss = np.mean(LOSS)
    auc = np.mean(AUC)
    return loss, auc




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






