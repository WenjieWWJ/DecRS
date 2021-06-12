import os
import time
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import util

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
    

data_path = '../../../data/amazon_book/'
item_category = np.load(data_path+'item_category.npy', allow_pickle=True).item()
category_list = np.load(data_path+'category_list.npy', allow_pickle=True).tolist()
training_list = np.load(data_path+'training_list.npy', allow_pickle=True).tolist()
valid_dict = np.load(data_path+'validation_dict.npy', allow_pickle=True).item()
test_dict = np.load(data_path+'testing_dict.npy', allow_pickle=True).item()
item_feature = np.load(data_path+'item_feature_file.npy', allow_pickle=True).item()
item_idx = list(item_feature.keys())

user_candidates_path = data_path + 'user_candidates_1k_mf.npy'
user_candidates = np.load(user_candidates_path, allow_pickle=True).item()

FM_user_item_score = np.load('./FM_user_item_scores.npy', allow_pickle=True).item()
NFM_user_item_score = np.load('./NFM_user_item_scores.npy', allow_pickle=True).item()
DecFM_user_item_score = np.load('./DecNFM_user_item_scores_0.005lr_2048bs_[0.2,0.2]dropout_0.0alpha_0.2lamda.npy', allow_pickle=True).item()


topN = [10, 20]

train_dict = {}
for pair in training_list:
    userID, itemID = pair
    if userID not in train_dict:
        train_dict[userID] = []
    train_dict[userID].append(itemID)

user_kl_score = {}
max_kl_score = 0
min_kl_score = 999
for userID in FM_user_item_score:
    train_set = train_dict[userID]
    valid_set = valid_dict[userID]
    kl_score = util.user_kl_score(train_set, valid_set, item_category, category_list)
    user_kl_score[userID]=kl_score
    if kl_score>max_kl_score:
        max_kl_score = kl_score
    if kl_score<min_kl_score:
        min_kl_score = kl_score
print(min_kl_score)
print(max_kl_score)


print("NFM")
threshold_list = [0, 0.5, 1, 2, 3, 4]
for threshold in threshold_list:
    user_item_pred = []
    user_item_gt = []
    print(f'threshold {threshold}')
    for userID in FM_user_item_score:
        if user_kl_score[userID]<=threshold:
            continue        
        NFM_item_scores = NFM_user_item_score[userID]
        DecFM_scores = []
        for i in range(len(NFM_item_scores)):
            itemID = user_candidates[userID][i]
            score = NFM_item_scores[i]
            DecFM_scores.append([itemID, score])
        DecFM_scores.sort(reverse=True, key=util.sort_function)

        user_item_gt.append(test_dict[userID])
        user_item_pred.append([x[0] for x in DecFM_scores[:1000]])
    print(f'user num {len(user_item_gt)}')
    test_result = util.computeTopNAccuracy(user_item_gt, user_item_pred, topN)
    util.print_results(None, None, test_result, None)
    
    
# print("DecNFM FM")
# threshold_list = [0, 0.5, 1, 2, 3, 4]
# alpha_list = [1]
# # alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# for threshold in threshold_list:
#     print(f'threshold {threshold}')
#     for alpha in alpha_list:
#         print(f'alpha {alpha}')
#         user_item_pred = []
#         user_item_gt = []
#         for userID in FM_user_item_score:
#             if user_kl_score[userID]<=threshold:
#                 continue
#             FM_item_scores = np.array(FM_user_item_score[userID])
#             DecFM_item_scores =  np.array(DecFM_user_item_score[userID])
#             lamda = ((user_kl_score[userID] - min_kl_score)/(max_kl_score-min_kl_score))**alpha

#             combined_scores = lamda * util.sigmoid(DecFM_item_scores) + (1-lamda) * util.sigmoid(FM_item_scores)
#             DecFM_scores = []
#             for i in range(len(FM_item_scores)): 
#                 itemID = user_candidates[userID][i]
#                 DecFM_scores.append([itemID, combined_scores[i]])
#             DecFM_scores.sort(reverse=True, key=util.sort_function)

#             user_item_gt.append(test_dict[userID])
#             user_item_pred.append([x[0] for x in DecFM_scores[:1000]])
#         print(f'user num {len(user_item_gt)}')
#         test_result = util.computeTopNAccuracy(user_item_gt, user_item_pred, topN)
#         util.print_results(None, None, test_result, None)

print("DecNFM.")
threshold_list = [0, 0.5, 1, 2, 3, 4]
alpha_list = [1]
# alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for threshold in threshold_list:
    print(f'threshold {threshold}')
    for alpha in alpha_list:
        print(f'alpha {alpha}')
        user_item_pred = []
        user_item_gt = []
        for userID in FM_user_item_score:
            if user_kl_score[userID]<=threshold:
                continue
            NFM_item_scores = np.array(NFM_user_item_score[userID])
            DecFM_item_scores =  np.array(DecFM_user_item_score[userID])
            lamda = ((user_kl_score[userID] - min_kl_score)/(max_kl_score-min_kl_score))**alpha

            combined_scores = lamda * util.sigmoid(DecFM_item_scores) + (1-lamda) * util.sigmoid(NFM_item_scores)
            DecFM_scores = []
            for i in range(len(NFM_item_scores)): 
                itemID = user_candidates[userID][i]
                DecFM_scores.append([itemID, combined_scores[i]])
            DecFM_scores.sort(reverse=True, key=util.sort_function)

            user_item_gt.append(test_dict[userID])
            user_item_pred.append([x[0] for x in DecFM_scores[:1000]])
        print(f'user num {len(user_item_gt)}')
        test_result = util.computeTopNAccuracy(user_item_gt, user_item_pred, topN)
        util.print_results(None, None, test_result, None)

