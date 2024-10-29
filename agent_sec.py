import numpy as np
import matplotlib.pyplot as plt
import sys
import random

sys.path.append('./models/')
#sys.path.insert(0, "/root/capsule/code/sec/models/")
from RandomReactiveLayer import ReactiveLayer as RL
from PerceptualLayer import PerceptualLayer as PL
from ContextualLayer_SEC import ContextualLayer as CL

class Agent(object):

    def __init__(self, episodes=5000, epsilon=1, selection='default', rec_thr=0.005,
        p_len=20, frozen_ws=False,
        stm_len=50, ltm_len=500,
        d_ine=True, forget="NONE", load_ltm=False,
        softmax=False, value_function='default'):

        self.RL = RL()
        self.PL = PL(prototype_length=p_len, frozen_weights=frozen_ws)
        self.CL = CL(pl=p_len, stm=stm_len, ltm=ltm_len,
            decision_inertia=d_ine, forget=forget, load_ltm=load_ltm,
            softmax=softmax, value_function=value_function)

        self.total_episodes = episodes
        self.epsilon = epsilon
        self.action_selection = selection
        print('Action selection mode: ', self.action_selection)
        self.reconstruct_thres = rec_thr # default 0.01 (originally 0.001)  --  best loss achieved 0.0024
        self.memory_threshold = 4 # number of memories required to activate the CL
        self.previous_couplet = np.array([np.zeros(p_len), np.zeros(2)])
        #self.previous_speed = np.zeros(3)
        self.layer_chosen = 'R' #Reactive Layer default selection
        self.count_reliable = 0
        self.count_unreliable = 0

    def reset(self, t=250):
        pass

    def step(self, obs, speed, reward, done, info):
        # MEMORY UPDATE PHASE
        # Update STM and LTM based on previous (state,action) couplet and current reward
        self.CL.update(self.previous_couplet, reward)

        # PERCEPTUAL UPDATE PHASE
        # Update Perceptual layer and obtain current prototype
        prototype = self.PL.get_prototype(obs)
        #print ('Current Protoype ', prototype) # length = 20 

        # Verify the quality of the current generated prototype
        # NOTE!!!: reconstruct_error is updated outside (at episodic_chamber.py LINE 131) at every step
        reconstruct_error = self.PL.get_reconstruct_error()
        #print ('Reconstruction Error: ', reconstruct_error)
        if reconstruct_error < self.reconstruct_thres: 
            self.count_reliable +=1  
        else:
            self.count_unreliable +=1

        if self.action_selection == 'default':
            action = self.dac_step(reconstruct_error, prototype)
        if self.action_selection == 'e-greedy':
            action = self.e_greedy_step(prototype)

        # Store couplet for next update of STM and LTM based on next reward.
        self.previous_couplet = [prototype, action]
        #self.previous_speed = speed

        return action


    def dac_step(self, reconstruct_error, prototype):
        # ACTION SELECTION PHASE
        action = np.array([0, 0])
        # Action proposed by the reactive layer.
        action_RL = self.RL.random_step()
        # Action proposed by the contextual layer.
        action_CL = self.CL.advance(prototype, argmax=False)

        # Arbitration procedure for final action selection (CL > RL).
        # (Chose CL action if the reconstruction error (ie. discrepancy) is smaller than the threshold value AND if there is content in the CL's memory)
        if reconstruct_error < self.reconstruct_thres and np.sum(action_CL) >= 0 and len(self.CL.LTM[2]) > self.memory_threshold:
            action = action_CL
            self.layer_chosen = 'C'
            #print ('Contextual ', action)
        else:
            action = action_RL
            self.layer_chosen = 'R'
            #print ('Reactive ', action)

        return action

    '''def dac_step(self, reconstruct_error, prototype):
        # Arbitration procedure for final action selection (CL > RL).
        # (Chose CL action if the reconstruction error (ie. discrepancy) is smaller than the threshold value AND if there is content in the CL's memory)
        if reconstruct_error < self.reconstruct_thres and len(self.CL.LTM[2]) > self.memory_threshold:
            # Action proposed by the contextual layer.
            action_CL = self.CL.advance(prototype, argmax=False)
            if np.sum(action_CL) >= 0:
                action = action_CL
                self.layer_chosen = 'C'
            else:
                action = self.RL.random_step()
                self.layer_chosen = 'R'
        else:
            # Action proposed by the reactive layer.
            action = self.RL.random_step()
            self.layer_chosen = 'R'

        return action'''

    def e_greedy_step(self, prototype):
        if (random.random() < self.epsilon):
            action = self.RL.random_step()
        else:
            action = self.CL.advance(prototype, argmax=True)

        return action

    def update_epsilon(self):
        if self.epsilon > 0.1: #R
            self.epsilon -= (1/self.total_episodes)

    def update_vision(self, obs):
        self.PL.advance(obs)
        #print ('Reconstruction Error: ', self.PL.reconstruct_error)
