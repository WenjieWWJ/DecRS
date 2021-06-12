import torch
import torch.nn as nn
import torch.nn.functional as F

class DecFM(nn.Module):
    def __init__(self, num_features, num_groups, num_factors, batch_norm, \
                 drop_prob, num_user_features, confounder_prior=None):
        super(DecFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.num_groups = num_groups
        self.num_user_features = num_user_features
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))
        self.confounder_embed = nn.Embedding(num_groups, num_factors)

        if confounder_prior is not None:
            # \bar{d} in the paper
            self.confounder_prior = torch.tensor(confounder_prior, dtype=torch.float32).cuda().unsqueeze(dim=-1)
        else:
            # confounder prior is assumed as [1/n, 1/n, ..., 1/n]
            self.confounder_prior = \
                    torch.tensor([1.0/num_groups for x in range(num_groups)]).cuda().unsqueeze(dim=-1)

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)


    def forward(self, features, feature_values):
        
        ### U & D => M: use FM to calculate M(\bar{d}, u)
        # map the id of group features. eg., [123, 124, 125] => [0, 1, 2]
        confounder_part = features[:, -self.num_groups:] - torch.min(features[:, -self.num_groups:])
        confounder_embed = self.confounder_embed(confounder_part) # N*num_confoudners*hidden_factor
        weighted_confounder_embed = confounder_embed * self.confounder_prior
        
        user_features = features[:, :self.num_user_features]
        user_feature_values = feature_values[:, :self.num_user_features].unsqueeze(dim=-1)
        user_features = self.embeddings(user_features)
        weighted_user_features = user_features * user_feature_values # N*num_user_features*hidden_factor
        
        user_confounder_embed = torch.cat([weighted_user_features, weighted_confounder_embed], 1)
        # FM model
        sum_square_user_confounder_embed = user_confounder_embed.sum(dim=1).pow(2) # N*hidden_factor
        square_sum_user_confounder_embed = (user_confounder_embed.pow(2)).sum(dim=1) # N*hidden_factor
        user_confounder_mediator = 0.5 * (sum_square_user_confounder_embed - square_sum_user_confounder_embed)
        user_confounder_mediator = user_confounder_mediator.unsqueeze(dim=1)
        
        batch_num = user_features.size()[0]
        assert list(user_confounder_mediator.size())==[batch_num, 1, self.num_factors]
        
        ### U & M & I => Y: similar to FM
        ## U I features
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values
        
        # M is used as one additional feature
        nonzero_embed = torch.cat([nonzero_embed, user_confounder_mediator], 1)

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)
        
        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)
    

class DecNFM(nn.Module):
    def __init__(self, num_features, num_groups, num_factors, \
                 act_function, layers, batch_norm, drop_prob, num_user_features=5, confounder_prior=None):
        super(DecNFM, self).__init__()
        self.num_features = num_features # include the confounder features
        self.num_groups = num_groups
        self.num_factors = num_factors
        self.num_user_features = num_user_features
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        if confounder_prior is not None:
            self.confounder_prior = torch.tensor(confounder_prior, dtype=torch.float32).cuda().unsqueeze(dim=-1)
        else:
            self.confounder_prior = \
                    torch.tensor([1.0/num_groups for x in range(num_groups)]).cuda().unsqueeze(dim=-1)
            
        self.embeddings = nn.Embedding(num_features, num_factors)
        self.confounder_embed = nn.Embedding(num_groups, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)
        
        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())
            MLP_module.append(nn.Dropout(drop_prob[-1]))

        self.deep_layers = nn.Sequential(*MLP_module)

        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)
        
        self._init_weight_()
        
    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)    
            
    def forward(self, features, feature_values):
        
        ### U & D => M: use FM to calculate M(\bar{d}, u)
        # map the id of group features. eg., [123, 124, 125] => [0, 1, 2]
        confounder_part = features[:, -self.num_groups:] - torch.min(features[:, -self.num_groups:])
        confounder_embed = self.confounder_embed(confounder_part) # N*num_confoudners*hidden_factor
        weighted_confounder_embed = confounder_embed * self.confounder_prior
        
        user_features = features[:, :self.num_user_features]
        user_feature_values = feature_values[:, :self.num_user_features].unsqueeze(dim=-1)
        user_features = self.embeddings(user_features)
        weighted_user_features = user_features * user_feature_values # N*num_user_features*hidden_factor
        
        user_confounder_embed = torch.cat([weighted_user_features, weighted_confounder_embed], 1)
        # FM model
        sum_square_user_confounder_embed = user_confounder_embed.sum(dim=1).pow(2) # N*hidden_factor
        square_sum_user_confounder_embed = (user_confounder_embed.pow(2)).sum(dim=1) # N*hidden_factor
        user_confounder_mediator = 0.5 * (sum_square_user_confounder_embed - square_sum_user_confounder_embed)
        user_confounder_mediator = user_confounder_mediator.unsqueeze(dim=1)
        
        batch_num = user_features.size()[0]
        assert list(user_confounder_mediator.size())==[batch_num, 1, self.num_factors]
        
        ### U & M & I => Y: similar to FM
        ## U I features
        features_embed = self.embeddings(features) # N*feature_num*hidden_factor
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = features_embed * feature_values # N*feature_num*hidden_factor
        
        # M is used as one additional feature
        nonzero_embed = torch.cat([nonzero_embed, user_confounder_mediator], 1)

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2) # N*hidden_factor
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1) # N*hidden_factor
        
        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)    
    

class FM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob):
        super(FM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)


    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)
        
        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)


class NoCFFM(nn.Module):
    def __init__(self, num_features, num_groups, num_factors, batch_norm, drop_prob):
        super(NoCFFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.num_groups = num_groups

        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)


    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features[:, :-self.num_groups])
        feature_values = feature_values[:, :-self.num_groups].unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)
        
        # bias addition
        feature_bias = self.biases(features[:, :-self.num_groups])
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)
    
    
class NFM(nn.Module):
    def __init__(self, num_features, num_factors, 
        act_function, layers, batch_norm, drop_prob, pretrain_FM):
        super(NFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)

        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(
                            self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(
                            self.pretrain_FM.biases.weight)
            self.bias_.data.copy_(self.pretrain_FM.bias_)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)


class NoCFNFM(nn.Module):
    def __init__(self, num_features, num_groups, num_factors, \
                 act_function, layers, batch_norm, drop_prob):
        super(NoCFNFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features # include the confounder features
        self.num_groups = num_groups
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)
        
        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())
            MLP_module.append(nn.Dropout(drop_prob[-1]))

        self.deep_layers = nn.Sequential(*MLP_module)

        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()
        
    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)
            
            
    def forward(self, features, feature_values):
        features_part = features[:, :-self.num_groups]
        feature_values_part = feature_values[:, :-self.num_groups].unsqueeze(dim=-1)
        features_embed = self.embeddings(features_part) 
        nonzero_embed = features_embed * feature_values_part 
                
        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2) # N*hidden_factor
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1) # N*hidden_factor
        
        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

        # bias addition
        feature_bias = self.biases(features_part)
        feature_bias = (feature_bias * feature_values_part).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)   

    

