import parl
import torch
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Agent, self).__init__(algorithm)
        self.act_dim = act_dim
        self.global_step = 0
        self.update_target_steps = 10
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
    
    def sample(self, obs):
        act = self.predict(obs)
        for i in range(self.alg.model.n_clusters):
            for j in range(len(self.act_dim)):
                if np.random.random() < self.e_greed:
                    act[i][j] = np.random.randint(self.act_dim[j])
                else:
                    if np.random.random() < 0.01:
                        act[i][j] = np.random.randint(self.act_dim[j])
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.alg.predict(obs)
        act = np.zeros((self.alg.model.n_clusters, len(self.act_dim)), dtype=np.int64)
        for i in range(self.alg.model.n_clusters):
            for j in range(len(self.act_dim)):
                act[i][j] = torch.argmax(logits[i][j]).numpy()
        return act
    
    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 1:
            self.alg.sync_target()
        self.global_step += 1

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss