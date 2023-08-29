import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import parl
from parl.utils.utils import check_model_method
import numpy as np

__all__ = ['DQN']


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None, max_grad_norm=0.5):
        self.model = model
        self.target_model = copy.deepcopy(model)
        device = torch.device("cpu")
        self.model.to(device)
        self.target_model.to(device)

        self.gamma = gamma
        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = [optim.Adam(self.model.net[i].parameters(), lr=self.lr, weight_decay=0.1) for i in range(self.model.n_clusters)]

    def predict(self, obs):
        pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        """
        obs : (n_clusters, batch_size, obs_dim)
        action : (n_clusters, batch_size, act_dim)
        logits : (n_clusters, act_dim, batch_size, n_actions)
        """
        L = []
        for i in range(self.model.n_clusters):
            logits = self.model.net[i](obs[i]) # [[batch_size, ni], [batch_size, nj] .. ]
            act = action[i].transpose(0, 1)
            pred_value = [logit.gather(1, act.unsqueeze(1)).reshape(-1) for logit, act in zip(logits, act)]
            pred_value = torch.stack(pred_value)
            Maxv = self.target_model.net[i](next_obs[i])
            with torch.no_grad():
                max_v = [maxv.max(1, keepdim=True)[0].reshape(-1) for maxv in Maxv]
                max_v = torch.stack(max_v)
                target = reward[i] + (1 - terminal[i]) * self.gamma * max_v
            self.optimizer[i].zero_grad()
            loss = self.mse_loss(pred_value, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.net[i].parameters(), self.max_grad_norm)
            self.optimizer[i].step()
            L.append(loss.item())
        return np.mean(L)


    def sync_target(self):
        self.model.sync_weights_to(self.target_model)