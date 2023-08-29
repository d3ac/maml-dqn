import os
import sys
if sys.platform == 'win32':
    sys.path.append(os.path.expanduser('C:/Users/10485/Desktop/科研训练/uavenv'))
else:
    sys.path.append(os.path.expanduser('~/Desktop/科研训练/uav env'))
from UAVenv.uav.uav import systemEnv
os.environ['PARL_BACKEND'] = 'torch'
import warnings

import gym
import numpy as np
import parl
import argparse
from parl.utils import logger, ReplayMemory
import pandas as pd

from cartpole_model import uavModel
from cartpole_agent import Agent
from env_utils import Wapper
from algorithm import DQN

LEARN_FREQ = 10  # training frequency
MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = 1e3
BATCH_SIZE = 64
LEARNING_RATE = 0.01
GAMMA = 0.99

def task_generator():# 生成seed
    task = np.random.randint(0, 1e9)
    return task

# train an episode
def run_train_episode(agent, env, rpm, task):
    total_reward = 0
    obs, _ = env.reset(task)
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        for i in range(env.n_clusters):
            rpm[i].append(obs[i], action[i], reward[i], next_obs[i], done[i])
        # train model
        if (len(rpm[i]) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = [], [], [], [], []
            for i in range(env.n_clusters):
                _obs, _act, _rew, _next_obs, _done = rpm[i].sample_batch(BATCH_SIZE)
                
                batch_obs.append(_obs)
                batch_action.append(_act)
                batch_reward.append(_rew)
                batch_next_obs.append(_next_obs)
                batch_done.append(_done)

            batch_obs = np.array(batch_obs)
            batch_action = np.array(batch_action)
            batch_reward = np.array(batch_reward)
            batch_next_obs = np.array(batch_next_obs)
            batch_done = np.array(batch_done)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done.all():
            break
    return total_reward


# evaluate 1 episodes
def run_evaluate_episodes(env, agent, eval_episodes, task):
    eval_reward = []
    for i in range(eval_episodes):
        obs, _ = env.reset(task)
        r = []
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            r.append(reward)
            if done.all():
                break
        eval_reward.append(np.mean(np.sum(np.array(r), axis=0)))
    return np.mean(eval_reward)


def main():
    env = Wapper(systemEnv())
    eval_env = Wapper(systemEnv())

    obs_dim = eval_env.obs_space
    act_dim = eval_env.act_space

    rpm = [ReplayMemory(MEMORY_SIZE, obs_dim[0], len(act_dim)) for _ in range(env.n_clusters)]

    # build an agent
    model = uavModel(obs_dim, act_dim, env.n_clusters)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-7)

    # warmup memory
    while len(rpm[0]) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm, task_generator())
    
    max_episode = args.max_episode

    # start training
    episode = 0
    data = []
    seed_pre = pd.read_csv('seed.csv', header=None)
    seed_set = set(seed_pre.values.reshape(-1).tolist())
    while episode < max_episode:
        # train part
        # for i in range(50):
        task = task_generator()
        params = model.get_params()
        total_reward, params = run_train_episode(agent, env, rpm, task)
        episode += 1

        # test part
        eval_reward = run_evaluate_episodes(eval_env, agent, 1, task)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(episode, agent.e_greed, eval_reward))
        data.append(eval_reward)
        Temp = pd.DataFrame(data)
        Temp.to_csv('data.csv', index=False, header=False)

        # save seed
        if eval_reward > 350:
            seed_set.add(task)
            pd.DataFrame(list(seed_set)).to_csv('seed.csv', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episode', type=int, default=300000,)
    args = parser.parse_args()

    main()