import numpy as np

import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from A_parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData

device = 'cuda:' + args.cuda
# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q

# load data
# path = 'data/' + args.data + '/'
path = '../dataset/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)  #这段代码的作用是将一个稀疏矩阵 train 中的非零元素转换为 1.0，并将零元素转换为 0.0，然后将其转换为 np.float32 类型。
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)

print(train)
print(test)
print('Data loaded.')
print(train.shape)
print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix 归一化
rowD = np.array(train.sum(1)).squeeze() # 计算的是每一行的非零元素的和。
colD = np.array(train.sum(0)).squeeze() # 计算的是每一列的非零元素的和。
#
# print(rowD,colD)
# print(len(train.data))
# train.data 存储的是稀疏矩阵中值为1的位置

for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo() # train.tocoo()：这是将 train 稀疏矩阵转换为 COO 格式（坐标格式）。
train_data = TrnData(train) #这里使用了一个自定义的类 TrnData 来封装转换后的 train 稀疏矩阵。
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0) # 表示在加载数据时使用的子进程数量。num_workers=0 表示不使用额外的子进程，即数据加载过程在主进程中进行。
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)  # 这段代码调用了一个函数（假设它是你代码中的自定义函数），它的作用是将 train 稀疏矩阵（通常是 scipy.sparse 格式）转换为 PyTorch 的稀疏张量（torch.sparse 格式）。


adj_norm = adj_norm.coalesce().cuda(torch.device(device))

# adj_norm 是一个张量使用 torch.sparse_coo（稀疏 COO 格式）表示，仅存储非零元素的索引和值。

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))


print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
print("adj =",adj,"q= ",svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
print(svd_u,s,svd_v)
del s
print('SVD done.')
# print(u_mul_s.shape)
# print(v_mul_s.shape)



# process test set
test_labels = [[] for i in range(test.shape[0])]  # 这行代码的作用是创建一个嵌套列表，其中包含与 test 数组（或 DataFrame）行数相同的空列表。
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]  # test.data 相当于一个稀疏矩阵然后 （row，col）就相当于非0元素的位置
    test_labels[row].append(col)  #test_labels 是一个嵌套列表 然后其中第i个位置就相当于第i个mashup，然后里面的数据就代表着此mashuo所包含的api的编号
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
# print(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr
total_recall = 0
total_ndcg = 0
total_precision = 0
total_map  = 0
test_time = 0
for epoch in range(epoch_no):
    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        # print("batch = ",batch)
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        #print(uids.shape,pos.shape,neg.shape)
        iids = torch.concat([pos, neg], dim=0)  # 将正样本 pos 和负样本 neg 拼接，生成一个包含所有正负样本的张量 iids
        # print(uids.type,uids.dtype,pos.type,pos.dtype,neg.type,neg.dtype)
        # print(uids,uids.shape,pos,pos.shape,neg,neg.shape,iids.shape)
        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        #print('batch',batch)
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        #print(i, len(train_loader), end='\r')
    # print(model.E_u,model.E_i)
    batch_no = len(train_loader) # batch_size的大小为4096然后train_data有6497条数据那么等于说只有两个batch
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
  #  print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if  (epoch+1) % 3 == 0:  # test every 10 epochs
        test_time +=1
        test_uids = np.array([i for i in range(adj_norm.shape[0])])  # test_uids是一个list[0,1.....,2905]
        batch_no = int(np.ceil(len(test_uids)/batch_user))  # ceil:2906/256
        all_recall = 0
        all_ndcg = 0
        all_precision = 0
        all_MAP = 0
        for batch in tqdm(range(batch_no)):  # 就是每次取256个 第一次[0,256)  [256,512).....
            start = batch*batch_user
            # print("batch = " ,batch)
            end = min((batch+1)*batch_user,len(test_uids))
            # print("start = ",start,"end = ",end)
            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            # print("test_uids_input = ", test_uids_input )
            predictions = model(test_uids_input,None,None,None,test=True)
            # print("prediction = ",predictions,"shape = ",predictions.shape)
            predictions = np.array(predictions.cpu())

            #top@20
            # print("****",test_uids[start:end],predictions,test_labels)
            recall, ndcg,predictions,MAP= metrics(test_uids[start:end],predictions,10,test_labels)
            #top@40

            all_recall+=recall
            all_ndcg+=ndcg
            all_precision+=predictions
            all_MAP+=MAP
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('-------------------------------------------')
        total_recall += all_recall/batch_no
        total_ndcg += all_ndcg/batch_no
        total_precision += all_precision/batch_no
        total_map += all_MAP/batch_no
        print('Test of epoch',epoch,':','Recall:',all_recall/batch_no,'Ndcg@20:',all_ndcg/batch_no,"precision:",all_precision/batch_no,'MAP:',all_MAP/batch_no)
total_recall = total_recall/test_time
total_ndcg = total_ndcg/test_time
total_precision = total_precision/test_time
total_map = total_map/test_time
print("recall",total_recall,"ndcg",total_ndcg,"precision",total_precision,"f1_score",2*total_recall*total_precision/(total_recall+total_precision),"map",total_map)
# final test
# test_uids = np.array([i for i in range(adj_norm.shape[0])])
# batch_no = int(np.ceil(len(test_uids)/batch_user))
#
# all_recall_20 = 0
# all_ndcg_20 = 0
# all_recall_40 = 0
# all_ndcg_40 = 0
# for batch in range(batch_no):
#     start = batch*batch_user
#     end = min((batch+1)*batch_user,len(test_uids))
#
#     test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
#     predictions = model(test_uids_input,None,None,None,test=True)
#     predictions = np.array(predictions.cpu())
#
#     #top@20
#     recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
#     #top@40
#     recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)
#
#     all_recall_20+=recall_20
#     all_ndcg_20+=ndcg_20
#     all_recall_40+=recall_40
#     all_ndcg_40+=ndcg_40
#     #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
# print('-------------------------------------------')
# print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)


#找到最终的嵌入表示并记录到文件中
# 自己加的
# 放在训练和测试循环的最后，在所有 epoch 循环结束之后提取嵌入向量
# user_embeddings = model.E_u.detach().cpu().numpy()
# item_embeddings = model.E_i.detach().cpu().numpy()
#
# # 将嵌入向量转换为 DataFrame
# user_df = pd.DataFrame(user_embeddings)
# item_df = pd.DataFrame(item_embeddings)
#
# # 保存为 CSV 文件
# user_df.to_csv("embedding/mashup_embeddings.csv", index=False, header=False)
# item_df.to_csv("embedding/api_embeddings.csv", index=False, header=False)
# # # 将嵌入向量保存到文件进一步分析
# # np.save('user_embeddings.npy', user_embeddings)
# # np.save('item_embeddings.npy', item_embeddings)
# print("User and item embeddings have been extracted and saved.")
#
# recall_20_x.append('Final')
# recall_20_y.append(all_recall_20/batch_no)
# ndcg_20_y.append(all_ndcg_20/batch_no)
# recall_40_y.append(all_recall_40/batch_no)
# ndcg_40_y.append(all_ndcg_40/batch_no)
#
# metric = pd.DataFrame({
#     'epoch':recall_20_x,
#     'recall@20':recall_20_y,
#     'ndcg@20':ndcg_20_y,
#     'recall@40':recall_40_y,
#     'ndcg@40':ndcg_40_y
# })
# current_t = time.gmtime()
# metric.to_csv('log/result_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.csv')
#
# torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')
# torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')