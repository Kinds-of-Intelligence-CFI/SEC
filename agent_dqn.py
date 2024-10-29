import os
import sys
import torch
import random
import numpy as np
import copy
from collections import deque

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, MaxPooling2D

sys.path.append('./models/')
from RandomReactiveLayer import ReactiveLayer as RL


class Agent(object):

    def __init__(self, input_size, output_size,
        random_steps, egreedy_steps, eps_max, eps_min,
        lr, gamma, batch_size, mem_size,
        dqn_update_freq, target_update_freq):

        self.RL = RL()
        self.AL = DQN(input_space=input_size, action_space=output_size,
            random_steps=random_steps, egreedy_steps=egreedy_steps, eps_max=eps_max, eps_min=eps_min,
            lr= lr, gamma=gamma, batch_size=batch_size, mem_size=mem_size,
            update_freq=dqn_update_freq, sync_freq=target_update_freq)

        self.step_count = 0

    def reset(self, t=250):
        pass

    def step(self, obs):
        self.step_count += 1

        if (random.random() < self.AL.epsilon) or (self.step_count < self.AL.epsilon_random_steps):
            action = self.RL.random_step()
            #print("action RL ", action)
        else:
            action = self.AL.greedy_step(obs)
            #print("action AL ", action)

        return action

    def update_DQN(self):
        #print("REPLAY LENGTH: ", len(self.AL.replay))
        if len(self.AL.replay) >= self.AL.batch_size:
            self.AL.update(self.AL.sample_buffer())

    def store_exp(self, state, action, reward, next_state, done):
        self.AL.update_buffer(state, action, reward, next_state, done)



class DQN(object):

    def __init__(self, input_space=[84, 84, 3], action_space=9,
        random_steps=1250, egreedy_steps=250000, eps_max=1.0, eps_min=0.01,
        lr= 0.25e-3, gamma=0.99, batch_size=200, mem_size=1000,
        update_freq=4, sync_freq=500):

        #print("input space: ", input_space)
        #print("action space: ", action_space)

        input_dim = input_space
        ff_dim = 512
        action_dim = action_space

        self.mainDQN = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  #padding needed?
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64*7*7, ff_dim),  # check this 64
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, action_dim),
            #torch.nn.ReLU()
        )

        self.targetDQN = copy.deepcopy(self.mainDQN) #A
        self.targetDQN.load_state_dict(self.mainDQN.state_dict()) #B

        self.loss_fn = torch.nn.MSELoss()
        self.losses = []

        self.learning_rate = lr
        self.optimizer = torch.optim.RMSprop(self.mainDQN.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.Adam(self.mainDQN.parameters(), lr=self.learning_rate)

        self.gamma = gamma

        self.epsilon = eps_max  # Epsilon greedy parameter
        self.epsilon_min = eps_min  # Minimum epsilon greedy parameter: 0.1
        #self.eval_epsilon = eps_eval # Evaulation time epxloration rate: 0.5
        self.epsilon_decay = (eps_max - eps_min) # Linear decay for annealing

        # Number of frames to take random action and observe output
        self.epsilon_random_steps = random_steps # atari breakout: 50K FRAMES - 1.25K STEPS (FRAMESKIP 4): 5 episodes
        # Number of frames for exploration
        self.epsilon_greedy_steps = egreedy_steps # atari breakout: 1M FRAMES - 25K STEPS (FRAMESKIP 4): 100 episodes

        self.action_space = action_space
        self.animalai_action = [3,3]

        self.batch_size = batch_size #B
        self.mem_size = mem_size
        self.replay = deque(maxlen=self.mem_size) #C

        self.update_freq = update_freq
        self.sync_freq = sync_freq #A1

        self.entropy = 0.


    def step(self, state):
        #print('state shape:', state.shape)
        # Transpose the dimensions of the input state
        state = np.transpose(state, (0, 3, 1, 2))
        #print('state shape:', state.shape)

        state_tensor = torch.Tensor(state)
        #print('state_tensor shape:', state_tensor.shape)

        qval = self.mainDQN(state_tensor)
        qval_ = qval.data.numpy()
        self.compute_entropy(qval_)

        if (random.random() < self.epsilon):
            action_ = np.random.randint(0,self.action_space)
            action = [int(action_/self.animalai_action[0]), int(action_%self.animalai_action[1])]
        else:
            action_ = np.argmax(qval_)
            action = [int(action_/self.animalai_action[0]), int(action_%self.animalai_action[1])]

        return action

    def greedy_step(self, state):
        state = np.transpose(state, (0, 3, 1, 2))
        #print('state shape:', state.shape)

        state_tensor = torch.Tensor(state)
        #print('state_tensor shape:', state_tensor.shape)

        qval = self.mainDQN(state_tensor)
        qval_ = qval.data.numpy()
        self.compute_entropy(qval_)

        action_ = np.argmax(qval_)
        action = [int(action_/self.animalai_action[0]), int(action_%self.animalai_action[1])]

        return action

    '''def update_epsilon(self):
        if self.epsilon > 0.1: #R
            self.epsilon -= (1/self.total_episodes)
            #print('EPSILON: ', self.epsilon)'''

    def update_epsilon(self):
        # Decay probability of taking random action
        self.epsilon -= self.epsilon_decay / self.epsilon_greedy_steps
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def update(self, batch):
        #print('UPDATING CONTROL DQN')
        state_mem, action_mem, reward_mem, next_state_mem, done_mem = batch
        #minibatch = random.sample(self.replay, self.batch_size)

        state_mem = np.transpose(state_mem, (0, 3, 1, 2))
        state1_batch = torch.Tensor(state_mem)
        action_batch = torch.Tensor(action_mem)
        reward_batch = torch.Tensor(reward_mem)
        next_state_mem = np.transpose(next_state_mem, (0, 3, 1, 2))
        state2_batch = torch.Tensor(next_state_mem)
        done_batch = torch.Tensor(done_mem)

        Q1 = self.mainDQN(state1_batch)
        #print('DQN VALUE PREDICITON Qs: ', Q1)
        #print('DQN VALUE PREDICITON Qs: ', Q1.shape)
        with torch.no_grad():
            Q2 = self.targetDQN(state2_batch) #B1

        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(X, Y.detach())
        #print(loss.item())
        #clear_output(wait=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

    def update_target(self):
        print('UPDATING TARGET DQN')
        self.targetDQN.load_state_dict(self.mainDQN.state_dict())

    def compute_entropy(self, policy):
        # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
        q = np.ravel(policy)
        # sofmax function corrected for large numbers
        #q = np.exp(q - np.max(q)) / np.exp(q - np.max(q)).sum()
        q = np.exp(q) / np.sum(np.exp(q))
        #print ("POLICY: ", q)
        #print ("PROBS SUM: ", np.sum(q))
        qlog = np.log2(q)
        qlog = np.nan_to_num(qlog)
        qqlog = q*qlog
        qsum = -np.sum(qqlog)
        self.entropy = qsum
        #print ("ENTROPY: ", self.entropy)

    #Add the state, action, reward, next_state and done to the current episode memory
    def update_buffer(self, state, action, reward, next_state, done):
        exp = [state, action, reward, next_state, done]
        self.replay.append(exp)

    def sample_buffer(self):
        indices = np.random.randint(low=0, high=len(self.replay), size=self.batch_size)
        state_mem, action_mem, reward_mem, next_state_mem, done_mem = [], [], [], [], []

        for i in indices:
            state_mem.append(self.replay[i][0])
            action_mem.append(self.replay[i][1])
            reward_mem.append(self.replay[i][2])
            next_state_mem.append(self.replay[i][3])
            done_mem.append(self.replay[i][4])

        state_mem = np.squeeze(state_mem, axis=1) # to convert state_mem from (64, 1, 84, 84, 3) to (64, 84, 84, 3)
        next_state_mem = np.squeeze(next_state_mem, axis=1)

        return (np.array(state_mem), np.array(action_mem), np.array(reward_mem), np.array(next_state_mem), np.array(done_mem))

    def get_memory_length(self):
        mem_len = len(self.replay)
        print("REPLAY BUFFER LENGTH: ", mem_len)
        return mem_len
