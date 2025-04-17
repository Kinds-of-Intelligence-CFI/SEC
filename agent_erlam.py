import os
import sys
import torch
import random
import time
import numpy as np
import copy
from math import sqrt
from collections import deque
from scipy.spatial import KDTree

# calculating euclidean distance between vectors
from scipy.spatial.distance import euclidean

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, MaxPooling2D

sys.path.append('./DAC/')
sys.path.append('./models/')
from RandomReactiveLayer import ReactiveLayer as RL


class Agent(object):

    def __init__(self, gameID, input_size, output_size,
        random_steps, egreedy_steps, eps_max, eps_min,
        lr, gamma, lamb, batch_size, mem_size, forgetting,
        dqn_update_freq, target_update_freq, assoc_freq, threshold):

        self.RL = RL()
        self.AL = ERLAM_DQN(input_space=input_size, action_space=output_size,
            random_steps=random_steps, egreedy_steps=egreedy_steps, eps_max=eps_max, eps_min=eps_min,
            lr= lr, gamma=gamma, lambda_=lamb,
            update_freq=dqn_update_freq, sync_freq=target_update_freq)
        self.CL = AssociativeMemory(gameID=gameID, gamma=gamma, batch_size=batch_size, mem_size=mem_size, forgetting=forgetting, assoc_freq=assoc_freq, threshold=threshold)

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
        if len(self.CL.memory) >= self.CL.batch_size:
            self.AL.update(self.CL.sample_batch())

    def store_exp(self, state, action, reward, next_state, done):
        self.CL.add_exp(state, action, reward, next_state, done)

    def update_memory(self, episodes):
        #In reference to the paper, perform lines 17-21 of Algorithm 2
        self.CL.update_graph()

        #In reference to the paper, perform line 23 of Algorithm 2
        #Please note: the paper implies that graph is updated once ever x episodes,
        #- However we do it every episode because of the small size of our memory
        self.CL.value_propagation(episodes)


class ERLAM_DQN(object):

    def __init__(self, input_space=[84, 84, 3], action_space=9,
        random_steps=1250, egreedy_steps=250000, eps_max=1.0, eps_min=0.01,
        lr= 0.25e-3, gamma=0.99, lambda_=0.3,
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
        self.lambda_ = lambda_

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

        self.update_freq = update_freq
        self.sync_freq = sync_freq

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
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, graph_value_mem = batch
        #minibatch = random.sample(self.replay, self.batch_size)
        #print('GRAPH VALUE MEM shape: ', graph_value_mem.shape)
        #print('GRAPH VALUE MEM: ', graph_value_mem)
        state_mem = np.transpose(state_mem, (0, 3, 1, 2))
        state1_batch = torch.Tensor(state_mem)
        action_batch = torch.Tensor(action_mem)
        reward_batch = torch.Tensor(reward_mem)
        next_state_mem = np.transpose(next_state_mem, (0, 3, 1, 2))
        state2_batch = torch.Tensor(next_state_mem)
        done_batch = torch.Tensor(done_mem)
        graph_value_batch = torch.Tensor(graph_value_mem)
        #print('GRAPH VALUE BATCH shape: ', graph_value_batch.shape)
        #print('GRAPH VALUE BATCH: ', graph_value_batch)
        Q1 = self.mainDQN(state1_batch)
        #print('DQN VALUE PREDICITON Qs: ', Q1)
        #print('DQN VALUE PREDICITON Qs: ', Q1.shape)
        with torch.no_grad():
            Q2 = self.targetDQN(state2_batch) #B1

        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        #print('DQN VALUE PREDICITON Qs: ', X)
        #print('DQN VALUE X: ', X.shape)
        #print('DQN VALUE Y_detach: ', Y.detach().shape)
        #loss1 = self.loss_fn(X, Y.detach()) + self.lambda_*self.loss_fn(X, graph_value_batch)
        #print("loss1: ", loss1.item())
        loss = torch.sum(torch.square(torch.subtract(Y.detach(), X)) + self.lambda_ * torch.square(torch.subtract(graph_value_batch, X))) / Y.detach().shape[0]
        #print("loss: ", loss.item())
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

"""
A node in the memory graph
It stores the state, action, reward, next state and done value just like a normal rl memory would
- in addition, we store t (the index of the episode when this was created) and Rt (the actual value of being in this state)
"""
class GraphNode:
    def __init__(self, key, embedding, state, t, action, reward, Rt, next_state, done, next_states, lru_timestamp):
        self.id = key
        self.embedding = embedding
        self.state = state
        self.t = t
        self.action = action
        self.reward = reward
        self.done = done
        self.Rt = Rt
        self.next_state = next_state #The next state originally chosen by the actor, not embedded
        self.next_states = next_states
        self.lru_timestamp = lru_timestamp


"""
Manages the memory for the rl agent
"""
class AssociativeMemory:
    def __init__(self, gameID='doubleTmaze', gamma=0.99, batch_size=64, mem_size=1000000, forgetting=True, assoc_freq=50, threshold=0.02):

        self.gamma = gamma

        self.memory = {}
        self.max_memory = mem_size
        self.batch_size = batch_size
        self.forgetting = forgetting
        self.associative_frequency=assoc_freq

        self.e_state_mem, self.e_action_mem, self.e_reward_mem, self.e_next_state_mem, self.e_done_mem = [], [], [], [], []

        self.graph_nodes = 0
        #self.threshold = 1e-7  # DEFAULT ERLAM RPs dim 4: 1e-7
        self.threshold = threshold   ## DEFAULT SEC AE dim 20: 0.02
        self.converge_threshold = 1e-6
        self.kd_tree = None
        self.current_timestamp = 0
        self.similar_states_count = 0
        self.updated_states_history = []

        #filename = './data/autoencoders/trained/autoencoder_p'+str(20)+'.h5'
        filename = './data/autoencoders/'+gameID+'/autoencoder_p'+str(20)+'.h5'
        filepath = os.path.abspath(filename)

        if os.path.exists(filepath):
            from keras.src.legacy.saving import legacy_h5_format

            AE = legacy_h5_format.load_model_from_hdf5(filepath, custom_objects={'mse': 'mse'})
            #AE = load_model(filename)
            print('FILE '+str(filename)+' LOADED')
            #AE.summary()
            layer = AE.get_layer('dense_4')
            self.phi = Model(inputs=AE.inputs, outputs=layer.output)
        else:
            print('FILE DOES NOT EXIST')
            print('Failed to find file on filepath: ', filepath)

        #self.phi = Model(inputs=AE.inputs, outputs=AE.get_layer('dense_1').output)

    #Takes a state and embeds it
    def embed(self, state):
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        return self.phi(state)[0]

    #Add the state, action, reward, next_state and done to the current episode memory
    def add_exp(self, state, action, reward, next_state, done):
        self.e_state_mem.append(state), self.e_action_mem.append(action), self.e_reward_mem.append(reward), self.e_next_state_mem.append(next_state), self.e_done_mem.append(done)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def calculate_sec_distance(self, embedding1, embedding2):
        #distance = 1 - (np.sum(np.abs(embedding1 - embedding2), axis=2)) / len(embedding1)
        distance = np.sum(np.abs(embedding1 - embedding2)) / len(embedding1)
        return distance

    def calculate_euclidean_distance(self, embedding1, embedding2):
        # first, calculate which are the closest states in memory (based on Euclidean distance btw state and memories)
        #distances = np.sum(np.abs(state - Q_ac[0]), axis=2) / len(state)
        #distance = np.linalg.norm(embedding1 - embedding2, axis=1)
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance

    def retrieve(self, embedding):
        # Find the closest matching key to the input embedding
        closest_key = min(self.memory.keys(), key=lambda key: np.linalg.norm(embedding - self.memory[key]))
        return closest_key

    #In reference to the paper: Episodic Reinforcment learning with Associative Memory
    #This is equivalent to lines 17-21 of Algorithm 2
    def update_graph(self):
        #print('UPDATING MEMORIES...')
        Rt = 0
        previous_node = None

        embeddings = np.array([memory.embedding for memory in self.memory.values()])
        self.kdtree = KDTree(embeddings) if len(embeddings) > 0 else None

        for t in reversed(range(len(self.e_state_mem))):

            Rt = self.e_reward_mem[t] + self.gamma * Rt

            embedding = self.embed(self.e_state_mem[t])
            #embedding = self.e_state_mem[t]
            #print('embedding shape: ', embedding.shape)

            #key = hash(tuple([int(x) for x in embedding]))
            key = hash(embedding.numpy().tobytes())
            #key = hash(tuple([round(x, 2) for x in embedding]))
            #key = hash(tuple([x for x in embedding]))
            #print('KEY: ', key)

            if len(self.memory) > 0 and self.kdtree is not None:

                dist, ind = self.kdtree.query(embedding, k=1)
                #print ("dist: ", dist)
                #print ("ind: ", ind)

                if ind is not None:
                    closest_key = list(self.memory.keys())[ind]
                    if dist < self.threshold:
                        #print("CLOSEST KEY FOUND!")
                        self.similar_states_count += 1
                        key = closest_key

            if key in self.memory:
                # UPDATE Rt
                if Rt > self.memory[key].Rt: # UPDATE Q_g
                    self.memory[key].Rt = Rt
                    self.memory[key].lru_timestamp = self.current_timestamp
                    self.current_timestamp += 1
                    #print('memory Rt: ', Rt)
                if not previous_node == None:
                    self.memory[key].next_states[self.e_action_mem[t]] = previous_node
                previous_node = self.memory[key]
            else:
                # GENERATE A NEW NODE IF MEMORY IS NOT FULL
                #self.check_memory()
                if len(self.memory) < self.max_memory:
                    #print("ADDING NEW GRAPH NODE!")
                    #self.graph_nodes += 1
                    self.memory[key] = GraphNode(key, embedding, self.e_state_mem[t], t, self.e_action_mem[t], self.e_reward_mem[t], Rt, self.e_next_state_mem[t], self.e_done_mem[t], {}, self.current_timestamp)
                    self.current_timestamp += 1
                    if not previous_node == None:
                        self.memory[key].next_states[self.e_action_mem[t]] = previous_node
                    previous_node = self.memory[key]

                # REMOVE LESS ACCESSED NODE IF MEMORY IS FULL AND FORGETTING IS ACTIVATED
                if self.forgetting == True and len(self.memory) >= self.max_memory:
                    #pass #Should delete some memory, the method for deleting is not covered explicitly in the paper and is therefore left open-ended here
                    # remove node with lowest LRU timestamp
                    lru_key = min(self.memory, key=lambda x: self.memory[x].lru_timestamp)
                    #print("lru_key:", lru_key)
                    self.memory.pop(lru_key)
                    #print("FORGETTING ACTIVATED!")
                    # remove node with lowest LRU timestamp - FROM THE NEXT STATE OF OTHER ELEMENTS!
                    for node in self.memory.values():
                        #node.next_states = {element for element in node.next_states if element.id != lru_key}
                        node.next_states = {element.id: element for element in node.next_states.values() if element.id != lru_key}

        self.e_state_mem, self.e_action_mem, self.e_reward_mem, self.e_next_state_mem, self.e_done_mem = [], [], [], [], []


    #In reference to the paper: Episodic Reinforcement learning with Associative Memory
    #This is equivalent to line: 23 of Algorithm 2
    def value_propagation(self, episodes): # VALUE PROPAGATION
        if episodes%self.associative_frequency == 0:
            print('UPDATING GRAPH...')
            converge = False
            prev_Rt = 10**7
            updated_states = 0

            mem_arr = list(self.memory.values())
            mem_arr.sort(reverse=True, key= lambda x: x.t)

            while not converge:
                #print('UPDATING GRAPH...')
                update_amount = 0
                for mem in mem_arr:
                    found = False
                    max_Rt = -9999999
                    for x in mem.next_states.values():
                        if x.Rt > max_Rt and not x == mem:
                            max_Rt = x.Rt
                            found = True
                            #print('FOUND!')
                    if found:
                        update_Rt = mem.reward + self.gamma * max_Rt
                        prev_Rt = mem.Rt
                        mem.Rt = update_Rt

                        diff = abs(update_Rt - prev_Rt)
                        update_amount += diff
                        updated_states += 1

                converge = update_amount <= self.converge_threshold

            self.updated_states_history.append(updated_states)
            #print('updated_states: ', updated_states)
            print('ERLAM CONVERGE!')


    #Samples a batch randomly from the memory
    def sample_batch(self):
        indices = np.random.randint(low=0, high=len(self.memory), size=self.batch_size)
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, graph_value_mem = [], [], [], [], [], []
        mem_arr = list(self.memory.values())

        for i in indices:
            if len(mem_arr[i].state.shape) == 3:
                mem_arr[i].state = np.expand_dims(mem_arr[i].state, axis=0)
            state_mem.append(mem_arr[i].state)
            action_mem.append(mem_arr[i].action)
            reward_mem.append(mem_arr[i].reward)
            next_state_mem.append(mem_arr[i].next_state)
            done_mem.append(mem_arr[i].done)
            if mem_arr[i].action in mem_arr[i].next_states:
                graph_value_mem.append(mem_arr[i].next_states[mem_arr[i].action].Rt)
            else:
                graph_value_mem.append(0)

        state_mem = np.squeeze(state_mem, axis=1) # to convert state_mem from (64, 1, 84, 84, 3) to (64, 84, 84, 3)
        next_state_mem = np.squeeze(next_state_mem, axis=1)

        return (np.array(state_mem), np.array(action_mem), np.array(reward_mem), np.array(next_state_mem), np.array(done_mem), np.array(graph_value_mem))

    def get_memory_length(self):
        #mem_arr = list(self.memory.values())
        mem_len = len(self.memory)
        #print("ASSOCIATIVE MEMORY LENGTH: ", mem_len)
        #print("GRAPH NODES: ", self.graph_nodes)
        print("GRAPH NODES: ", mem_len)
        return mem_len

    def estimate_return(self, state):
        Q_s = []
        q = -10
        self.k = 2

        embedding = self.embed(state)
        #state = state.tolist()
        key = hash(tuple([int(x) for x in embedding]))
        #print('KEY: ', key)

        if not key in self.memory:
            distances = np.linalg.norm(key - self.memory)
            # get an index of the closest neighbors
            #nearest_neighbor_ids = distances.argsort()[:self.k] if len(distances) > self.k else distances.argsort()
            nearest_neighbor_ids = distances.argsort()[:self.k]
            # get q values of closest neighbors
            k_qs = [Q_ac[1][k] for j, k in enumerate(nearest_neighbor_ids)]
            # get the q estimate for the new state
            q = np.mean(k_qs)
        else:
            Q_s = self.memory[key].Rt

        return Q_s
