import os
import math
import numpy as np

def kl_div(p_dis, q_dis, alpha=0.01):
    KL_res = 0
    for index, p_value in enumerate(p_dis):
        if p_value < 1e-5:
            continue
        q_value = (1-alpha) * q_dis[index] + alpha * p_dis[index]
        KL_res += p_value*(np.log(p_value/q_value))
    return KL_res

def user_kl_score(training_set, validation_set, item_category, category_list):
    items_list = training_set+validation_set
    items_len = len(items_list)
    training_set_1 = items_list[:items_len//2]
    training_set_2 = items_list[items_len//2:]
    
    training_set_1_dis = [0] * len(category_list)
    for itemID in training_set_1:
        categories = item_category[itemID]
        for cate in categories:
            training_set_1_dis[cate] += round(1.0/len(categories), 4)
    training_set_1_dis = [x/len(training_set_1) for x in training_set_1_dis]

    training_set_2_dis = [0] * len(category_list)
    for itemID in training_set_2:
        categories = item_category[itemID]
        for cate in categories:
            training_set_2_dis[cate] += round(1.0/len(categories), 4)
    training_set_2_dis = [x/len(training_set_2) for x in training_set_2_dis]
    
    kl_res_1 = kl_div(training_set_1_dis, training_set_2_dis)
    kl_res_2 = kl_div(training_set_2_dis, training_set_1_dis)
    
    return kl_res_1+kl_res_2

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sort_function(elem):
    return elem[1]

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR

def calibration(item_feature, test_dict, train_dict, user_pred, topN, cate_num):
    C_KL = []
    C_H = []
    C_E = []
    assert len(test_dict) == len(user_pred)
    
    alpha = 0.01 # refer to the setting in "calibrated recommendation"
    for i, userID in enumerate(test_dict):
        
        history_items = train_dict[userID]
        history_cate = np.array([0]*cate_num, dtype=np.float32)
        for itemID in history_items:
            history_cate += np.array(item_feature[itemID][1][-cate_num:], dtype=np.float32)
        history_cate = history_cate/len(history_items)
        
        C_KL_u = []
        C_H_u = []
        C_E_u = []
        for n in topN:
            rec_list = user_pred[i][:n]
            rec_cate = np.array([0]*cate_num, dtype=np.float32)
            for itemID in rec_list:
                rec_cate += np.array(item_feature[itemID][1][-cate_num:], dtype=np.float32)
            rec_cate = rec_cate/len(rec_list)
            C_KL_res = 0
            C_H_res = 0
            C_E_res = 0
            for j, p_cate in enumerate(history_cate):
                C_H_res += (np.sqrt(p_cate) - np.sqrt(rec_cate[j])) * (np.sqrt(p_cate) - np.sqrt(rec_cate[j]))
                C_E_res += (p_cate-rec_cate[j]) * (p_cate-rec_cate[j])
                if p_cate < 1e-5:
                    continue
                q_cate = (1-alpha) * rec_cate[j] + alpha * p_cate
                C_KL_res += p_cate*(np.log(p_cate/q_cate))
            C_KL_u.append(C_KL_res)
            C_H_u.append(np.sqrt(C_H_res)/np.sqrt(2))
            C_E_u.append(np.sqrt(C_E_res))
        C_KL.append(C_KL_u)
        C_H.append(C_H_u)
        C_E.append(C_E_u)
    C_KL = np.around(np.mean(C_KL, 0), 4).tolist()
    C_H = np.around(np.mean(C_H, 0), 4).tolist()
    C_E = np.around(np.mean(C_E, 0), 4).tolist()
    return C_KL, C_H, C_E

def print_results(train_RMSE, valid_result, test_result, calibration_results):
    """output the evaluation results."""
    if train_RMSE is not None:
        print("[Train]: RMSE: {:.4f}".format(train_RMSE))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

    if calibration_results is not None:
        print("[Calibration]: C_KL: {} C_H: {} C_E: {}".format(
                            '-'.join([str(x) for x in calibration_results[0]]), 
                            '-'.join([str(x) for x in calibration_results[1]]), 
                            '-'.join([str(x) for x in calibration_results[2]])))
        
        