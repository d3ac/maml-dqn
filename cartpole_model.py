import torch
import torch.nn as nn
import torch.nn.functional as F
import parl
from collections import OrderedDict
    
class baseModel(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(baseModel, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.ModuleList([nn.Linear(64, act_shape[i]) for i in range(len(act_shape))])
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, obs, params):
        obs = obs.to(torch.float32)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        logits = [self.fc_pi[i](obs) for i in range(len(self.fc_pi))]
        return logits


class uavModel(parl.Model):
    def __init__(self, obs_space, act_space, n_clusters):
        super(uavModel, self).__init__()
        self.net = nn.ModuleList([baseModel(obs_space, act_space) for i in range(n_clusters)])
        self.n_clusters = n_clusters
        self.n_act = len(act_space)
    
    def forward(self, obs, params):
        return [self.net[i].forward(obs[i].reshape(-1, obs.shape[-1]), params[i]) for i in range(len(self.net))]
    
    def get_params(self):
        return [OrderedDict(self.net[i].named_parameters()) for i in range(self.n_clusters)]