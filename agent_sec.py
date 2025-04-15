import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./DAC/')
sys.path.append('./models/')
from RandomReactiveLayer import ReactiveLayer as RL
from PerceptualLayer import PerceptualLayer as PL
from ContextualLayer import ContextualLayer as CL


class ContextualAgent(object):

    def __init__(self, stm_len, ltm_len, p_len=20, rec_thr=0.005, d_ine=True, forget=True, value_function='default'):
        self.RL = RL()
        self.PL = PL(prototype_length=p_len)
        self.CL = CL(stm=stm_len, ltm=ltm_len, pl=p_len, load_ltm=False, decision_inertia=d_ine, forget=forget, value_function=value_function)
        self.previous_couplet = [np.zeros(p_len), np.zeros(2)]
        self.previous_speed = np.zeros(3)
        self.reconstruct_thres = rec_thr # default 0.01 (originally 0.001)  --  best loss achieved 0.0024
        self.layer_chosen = 'R' #Reactive Layer default selection
        self.count_reliable = 0
        self.count_unreliable = 0
        self.memory_threshold = 4

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
        #print("state ", type(prototype))
        #print("state ", prototype.shape) # proportional to sequence's length, n = LTM sequences

        # Verify the quality of the current generated prototype
        # NOTE!!!: reconstruct_error is updated outside (at episodic_chamber.py LINE 131) at every step
        reconstruct_error = self.PL.get_reconstruct_error()
        #print ('Reconstruction Error: ', reconstruct_error)
        if reconstruct_error < self.reconstruct_thres: 
            self.count_reliable +=1  
        else:
            self.count_unreliable +=1
        
        # ACTION SELECTION PHASE
        action = np.array([0, 0])
        # Action proposed by the reactive layer.
        action_RL = self.RL.random_step()
        # Action proposed by the contextual layer.
        action_CL = self.CL.advance(prototype, reconstruct_error)

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

        # Store couplet for next update of STM and LTM based on next reward.
        self.previous_couplet = [prototype, action]
        self.previous_speed = speed

        return action

    def update_vision(self, obs):
        self.PL.advance(obs)
        #print ('Reconstruction Error: ', self.PL.reconstruct_error)
