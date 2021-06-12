import numpy as np
import torch.utils.data as data
import scipy.sparse as sp

def read_features(file_path, features):
    """ Read features from the given file. """
    i = len(features)
    file = np.load(file_path, allow_pickle=True).item()
    for ID in file:
        feat = file[ID][0]
        for f in feat:
            if f not in features:
                features[f] = i
                i += 1
    return file, features


def map_features(user_feature_path, item_feature_path):
    """ Get the number of existing features of users and items."""
    features = {}
    user_feature, features = read_features(user_feature_path, features)
    item_feature, features = read_features(item_feature_path, features)
    print("number of users: {}".format(len(user_feature)))
    print("number of items: {}".format(len(item_feature)))
    print("number of features: {}".format(len(features)))
    
    mapped_user_feature = {}
    for userID in user_feature:
        feat = user_feature[userID][0]
        mapped_feat = []
        for f in feat:
            mapped_feat.append(features[f])
        mapped_user_feature[userID] = [mapped_feat, user_feature[userID][1]]
    mapped_item_feature = {}
    for itemID in item_feature:
        feat = item_feature[itemID][0]
        mapped_feat = []
        for f in feat:
            mapped_feat.append(features[f])
        mapped_item_feature[itemID] = [mapped_feat, item_feature[itemID][1]]
    
    return mapped_user_feature, mapped_item_feature, len(features)


def loadData(path):
    """load data for testing."""
    file = np.load(path, allow_pickle=True).item()
    return file


class FMData(data.Dataset):
    """ Construct the FM training dataset."""
    def __init__(self, train_path, user_feature, item_feature, loss_type, valid_path=None, negative_num=1):
        super(FMData, self).__init__()
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.loss_type = loss_type
        self.negative_num = negative_num
        self.valid_path = valid_path

        # used for training
        self.pos_label = []
        self.pos_features = []
        self.pos_feature_values = []
        
        # used for negative sampling
        self.train_list = np.load(train_path, allow_pickle=True).tolist()
        self.train_dict = {}
        self.user_max_ID = max(user_feature.keys())
        self.item_max_ID = max(item_feature.keys())
        
        user_feature_list = []
        user_feature_values_list = []
        for userID in range(self.user_max_ID+1):
            user_feature_list.append(user_feature[userID][0])
            user_feature_values_list.append(user_feature[userID][1])

        item_feature_list = []
        item_feature_values_list = []
        for itemID in range(self.item_max_ID+1):
            item_feature_list.append(item_feature[itemID][0])
            item_feature_values_list.append(item_feature[itemID][1])

        self.item_feature_list = np.array(item_feature_list)
        self.item_feature_values_list = np.array(item_feature_values_list, np.float32)
        self.user_feature_list = np.array(user_feature_list)
        self.user_feature_values_list = np.array(user_feature_values_list, np.float32)
        
        self.user_item_mat = sp.dok_matrix((self.user_max_ID+1, self.item_max_ID+1), dtype=np.float32)
        for pair in self.train_list:
            userID, itemID = pair
            self.user_item_mat[userID, itemID] = 1
            if userID not in self.train_dict:
                self.train_dict[userID] = []
            self.train_dict[userID].append(itemID)
            
            if valid_path is None: # FMData for training data
                # features used for model training
                self.pos_features.append(np.concatenate((self.user_feature_list[userID],self.item_feature_list[itemID])))
                self.pos_feature_values.append(np.concatenate(\
                                            (self.user_feature_values_list[userID],self.item_feature_values_list[itemID])))
                if self.loss_type == 'square_loss':
                    self.pos_label.append(np.float32(1.0))
                else: # log_loss
                    self.pos_label.append(np.float32(1.0))
        
        if valid_path is not None: # FMData for validation data
            self.valid_list = np.load(valid_path, allow_pickle=True).tolist()
            for pair in self.valid_list:
                userID, itemID = pair
                self.user_item_mat[userID, itemID] = 1
                # features used for model training
                self.pos_features.append(np.concatenate((self.user_feature_list[userID],self.item_feature_list[itemID])))
                self.pos_feature_values.append(np.concatenate(\
                                            (self.user_feature_values_list[userID],self.item_feature_values_list[itemID])))
                if self.loss_type == 'square_loss':
                    self.pos_label.append(np.float32(1.0))
                else: # log_loss
                    self.pos_label.append(np.float32(1.0))

        self.label = self.pos_label
        self.features = self.pos_features
        self.feature_values = self.pos_feature_values
        
    def ng_sample(self):
        
        # negative sampling
        neg_label = []
        neg_features = []
        neg_feature_values = []
        item_num = len(self.item_feature)
        if self.valid_path is None: # training phase
            pair_list = self.train_list
        else: # validation phase
            pair_list = self.valid_list

        for pair in pair_list:
            userID, itemID = pair
            for i in range(self.negative_num):
                j = np.random.randint(item_num)
                while (userID, j) in self.user_item_mat:
                    j = np.random.randint(item_num)
                
                neg_features.append(np.concatenate((self.user_feature_list[userID],self.item_feature_list[j])))
                neg_feature_values.append(np.concatenate(
                                        (self.user_feature_values_list[userID],self.item_feature_values_list[j])))
                
                if self.loss_type == 'square_loss':
                    neg_label.append(np.float32(-1.0))
                else:
                    neg_label.append(np.float32(0))

        self.features = self.pos_features + neg_features
        self.feature_values = self.pos_feature_values + neg_feature_values
        self.label = self.pos_label + neg_label
        
        assert len(self.features) == len(self.feature_values) == len(self.label)
        assert all(len(item) == len(self.features[0]
            ) for item in self.features), 'features are of different length'

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        features = self.features[idx]
        feature_values = self.feature_values[idx]
        return features, feature_values, label
