import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import autograd
from torch.utils.data import TensorDataset
from Dataset import Dataset
from torch import optim
from __future__ import division
from multiprocessing import Pool
from multiprocessing import cpu_count
from torch import nn
import math
import torch.nn.functional as f

# ====================================
# こちらのコードを使用させていただいた。
# https://github.com/hexiangnan/adversarial_personalized_ranking/blob/master/AMF.py
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys():
        # positive instance
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos


def shuffle(samples, batch_size, dataset):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    
    global _dataset
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _dataset = dataset
    _index = range(len(_user_input))
    np.random.shuffle(_index)
    num_batch = len(_user_input) // _batch_size
    
    pool = Pool(cpu_count())
    res = pool.map(_get_train_batch, range(num_batch))
    pool.close()
    pool.join()
    
    user_list = [r[0] for r in res]
    item_pos_list = [r[1] for r in res]
    user_dns_list = [r[2] for r in res]
    item_dns_list = [r[3] for r in res]
    
    return user_list, item_pos_list, user_dns_list, item_dns_list


def _get_train_batch(i):
    user_batch, item_batch = [], []
    user_neg_batch, item_neg_batch = [], []
    
    begin = i * _batch_size
    
    for idx in range(begin, begin + _batch_size):
        
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input_pos[_index[idx]])
        
        for dns in range(1):
            
            user = _user_input[_index[idx]]
            user_neg_batch.append(user)
            
            # negtive k
            j = np.random.randint(_dataset.num_items)
            
            while j in _dataset.trainList[_user_input[_index[idx]]]:
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)
            
    return np.array(user_batch)[:, None], np.array(item_batch)[:, None], \
           np.array(user_neg_batch)[:, None], np.array(item_neg_batch)[:, None]


def init_eval_model():
    pool = Pool(cpu_count())
    feed_dicts = pool.map(evaluate_input, range(dataset.num_users))
    pool.close()
    pool.join()
    return feed_dicts


def evaluate_input(user):
    # generate items_list
    test_item = dataset.testRatings[user][1]
    item_input = set(range(dataset.num_items)) - set(dataset.trainList[user])
    if test_item in item_input:
        item_input.remove(test_item)
    item_input = list(item_input)
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:, None]
    return user_input, item_input
# ====================================

class APR(nn.Module):

    def __init__(self, n_user, n_item,  dataset, adv=1, feed_dicts=None, k=8):        
        super(APR, self).__init__()
        self.reg = 1 
        self.adv = adv
        self.reg_adv = 1
        self.eps = 0.5
        self.eval_data = _evaluate_input()
        
        self.embedding_P = nn.Embedding(n_user, k)#.from_pretrained(emb_P)
        self.embedding_Q = nn.Embedding(n_item, k)#.from_pretrained(emb_Q)
        nn.init.normal_(self.embedding_P.weight, 0.0, 0.01)
        nn.init.normal_(self.embedding_Q.weight, 0.0, 0.01)
        
        # adversarial
        self.delta_P = nn.Embedding(n_user, k)
        self.delta_Q = nn.Embedding(n_item, k)
        
        self.delta_P .weight.requires_grad = False
        self.delta_Q .weight.requires_grad = False
        
        self.delta_P.weight = nn.Parameter(torch.zeros([n_user, k], dtype=torch.float32))
        self.delta_Q.weight = nn.Parameter(torch.zeros([n_user, k], dtype=torch.float32))
        
        self.h = torch.ones(k, 1)
        self.dataset = dataset
        self.eval_data =  feed_dicts
    
    def forward(self, user, item):        
        if self.adv == 0:
            return torch.matmul(self.embedding_P(user.flatten()) * self.embedding_Q(item.flatten()), self.h).sum(1)
        else:
            P_plus_delta = torch.add(self.embedding_P(user.flatten()), self.delta_P(user.flatten()))
            Q_plus_delta = self.embedding_Q(user.flatten()) + self.delta_Q(user.flatten())
            adv_term = torch.matmul(P_plus_delta * Q_plus_delta, self.h).sum(1)            
            return torch.matmul(self.embedding_P(user.flatten()) * self.embedding_Q(item.flatten()), self.h).sum(1), adv_term 
                
    def bpr_loss(self, user, items):
        pos_preds = self._predict(user, items[0])
        neg_preds = self._predict(user, items[1])
        preds = pos_preds - neg_preds
        loss = torch.log(1 + torch.exp(-preds)).sum()
        return loss
    
    def apr_loss(self, user, items):
        pos_preds_adv = self._predict_adv(user, items[0])
        neg_preds_adv = self._predict_adv(user, items[1])
        preds_adv = pos_preds_adv - neg_preds_adv 
        loss = self.reg_adv * torch.log(1 + torch.exp(-preds_adv)).sum()  
        return loss
    
    def _predict(self, user, item):        
        return torch.matmul(self.embedding_P(user.flatten()) * self.embedding_Q(item.flatten()), self.h).sum(1)
    
    def _predict_adv(self, user, item):  
        P_plus_delta = torch.add(self.embedding_P(user.flatten()), self.delta_P(user.flatten()))
        Q_plus_delta = torch.add(self.embedding_Q(user.flatten()), self.delta_Q(user.flatten()))
        return torch.matmul(P_plus_delta * Q_plus_delta, self.h).sum(1)
                
    def init_eval_model():
        pool = Pool(cpu_count())
        feed_dicts = pool.map(_evaluate_input, range(dataset.num_users))
        pool.close()
        pool.join()
        return feed_dicts

    def _evaluate_input(user):
        # generate items_list
        test_item = dataset.testRatings[user][1]
        item_input = set(range(dataset.num_items)) - set(dataset.trainList[user])
        if test_item in item_input:
            item_input.remove(test_item)
        item_input = list(item_input)
        item_input.append(test_item)
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input

    def evaluate(self):
        res = []
        for user in range(self.dataset.num_users):
            res.append(self._eval_by_user(user))
        res = np.array(res)
        hr, ndcg, auc = (res.mean(axis=0)).tolist()
        return hr, ndcg, auc

    def _eval_by_user(self, user):

        user_input, item_input = self.eval_data[user]

        predictions = self.predict(torch.tensor(user_input, dtype=torch.long), torch.tensor(item_input, dtype=torch.long))
        neg_predict, pos_predict = predictions[:-1], predictions[-1]
        position = (neg_predict >= pos_predict).sum()
        hr, ndcg, auc = [], [], []
        for k in range(1,100 + 1):
            hr.append(position < k)
            ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
            auc.append(1 - (position / len(neg_predict)))

        return hr, ndcg, auc

# datasetは付きのレポジトリのデータを使用
# https://github.com/hexiangnan/adversarial_personalized_ranking

dataset = Dataset("Data/yelp")
samples = sampling(dataset)
train = shuffle(samples, 512, dataset)
train_dataset = TensorDataset(torch.tensor(train, dtype=torch.int64)
                              
adv = 1
n_user = dataset.num_users
n_item = dataset.num_items
model = APR(n_user, n_item, dataset, adv, feed_dicts=eval_d)
optimizer = optim.SGD(model.parameters(), lr=5e-2)

model.train()
for epoch in range(1000):
    print("epoch", epoch)
    for i in range(1312):
        
        model.zero_grad()

        user = autograd.Variable(train_dataset[0][0][i])
        item_pos = autograd.Variable(train_dataset[1][0][i])
        item_neg = autograd.Variable(train_dataset[3][0][i])
        
        user = user.view(-1,)
        item_pos = item_pos.view(-1,)
        item_neg = item_neg.view(-1,)        

        loss_bpr = model.bpr_loss(user, (item_pos, item_neg))
        loss_bpr.backward()
        
        grad_P = model.embedding_P.weight.grad
        grad_Q = model.embedding_Q.weight.grad
        model.delta_P.weight =  nn.Parameter(model.eps * f.normalize(model.embedding_P.weight.grad, p=2, dim=1))
        model.delta_Q.weight =  nn.Parameter(model.eps * f.normalize(model.embedding_Q.weight.grad, p=2, dim=1))
        
        loss_apr = model.apr_loss(user, (item_pos, item_neg))
        loss_apr.backward()
                              
        # Update the parameters
        optimizer.step()
        
    print(model.state_dict())
    print(loss)

print("eval")
result = model.evaluate()
hr, ndcg, auc = np.swapaxes(result, 0, 1)[-1]
print(hr, ndcg, auc)