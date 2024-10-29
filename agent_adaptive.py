import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('./models/')
#sys.path.insert(0, "/root/capsule/code/sec/models/")
from RandomReactiveLayer import ReactiveLayer as RL
from PerceptualLayer import PerceptualLayer as PL

class ReactiveAgent(object):

    def __init__(self, p_len=20, rec_thr=0.005):
        self.RL = RL()
        self.PL = PL(prototype_length=p_len)
        self.previous_couplet = np.array([np.zeros(p_len), np.zeros(2)])
        self.previous_speed = np.zeros(3)
        self.reconstruct_thres = rec_thr
        #self.reconstruct_error = 100
        self.layer_chosen = 'R'
        self.count_reliable = 0
        self.count_unreliable = 0

    def reset(self, t=250):
        pass

    def step(self, obs, speed, reward, done, info):

        # PERCEPTUAL UPDATE PHASE
        # Update Perceptual layer and obtain current prototype
        prototype = self.PL.get_prototype(obs)
        #print ('Current Protoype ', prototype) # length = 20 
        #print ('Current Protoype length ', len(prototype)) 

        # Verify the quality of the current generated prototype
        #NOTE!!!: reconstruct_error is updated outside (at episodic_chamber.py LINE 131) at every step
        reconstruct_error = self.PL.get_reconstruct_error()
        #print ('Reconstruction Error: ', reconstruct_error)
        if reconstruct_error < self.reconstruct_thres: 
            self.count_reliable +=1  
        else:
            self.count_unreliable +=1
            
        # ACTION SELECTION PHASE
        action = np.array([0, 0])

        # Action proposed by the reactive layer.
        #action_RL = self.RL.feed_img(obs)
        action_RL = self.RL.random_step()

        action = action_RL
        self.layer_chosen = 'R'
        #print ('Reactive ', action)

        self.previous_speed = speed

        return action

    def update_vision(self, obs):
        self.PL.advance(obs)
        #print ('Reconstruction Error: ', self.PL.reconstruct_error)
