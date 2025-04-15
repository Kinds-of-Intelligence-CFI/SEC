import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('./DAC/')
sys.path.append('./models/')
#sys.path.insert(0, "/root/capsule/code/sec/models/")
from RandomReactiveLayer import ReactiveLayer as RL
from PerceptualLayer import PerceptualLayer as PerceptualLayer_AE
from PerceptualLayer_RP_Gauss import PerceptualLayer_RP
from ContextualLayer_MFEC import MFEC


class Agent(object):

    def __init__(self, random_steps=1250, epsilon=0.005, discount=0.99, k=50, ltm_len=100, embbeding_type='random_projection', p_len=20, rec_thr=0.005, forget="NONE", estimation=False, frozen_ws=False, load_ltm=False):
        self.RL = RL()

        self.embbeding_type = embbeding_type
        if self.embbeding_type == 'autoencoder':
            print('ADAPTIVE LAYER AS AN AUTOENCODER')
            #self.PL = PerceptualLayer_AE(prototype_length=p_len, frozen_weights=frozen_ws)
            self.PL = PerceptualLayer_AE(prototype_length=p_len)
        if self.embbeding_type == 'random_projection':
            print('ADAPTIVE LAYER AS A RANDOM PROJECTION')
            self.PL = PerceptualLayer_RP(img_shape=(84,84,3), autoencoder=self.embbeding_type, prototype_length=p_len, frozen_weights=frozen_ws)

        self.CL = MFEC(discount=discount, k=k, ltm=ltm_len, pl=p_len, forget=forget, estimation=estimation, load_ltm=load_ltm)
        #self.CL = CL(stm=stm_len, ltm=ltm_len, pl=p_len, load_ltm=False, decision_inertia=d_ine, forget=forget)

        self.layer_chosen = 'R' #Reactive Layer default selection
        self.previous_couplet = np.array([np.zeros(p_len), 0]).tolist()

        self.action_space = [3, 3]
        self.episode_buffer = []
        self.reward_buffer = []

        self.epsilon_random_steps = random_steps
        self.epsilon=epsilon
        self.discount=discount

        self.reconstruct_thres = rec_thr # default 0.01 (originally 0.001)  --  best loss achieved 0.0024
        self.count_reliable = 0
        self.count_unreliable = 0
        self.rewards = 0
        self.step_count = 0
        self.memory_threshold = 5


    def reset(self, t=250):
        pass

    def step(self, obs, speed, reward, done, info):
        self.step_count += 1

        # PHASE 0. Store state-action-reward from previous step into the episode buffer
        self.reward_buffer.append(reward)
        self.episode_buffer.append(self.previous_couplet)

        # PHASE 1. Recieve observation from the environment (PERCEPTUAL UPDATE)

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
        
        # PHASE 2. Estimate return of each action via (paper equation 2)

        # PHASE 3. Action selection
        action = np.array([0, 0])
        # Action proposed by the reactive layer.
        #action_RL = self.RL.random_step()
        # Action proposed by the contextual layer.
        #action_CL = self.CL.action_selection(Q_s)

        if (np.random.random() < self.epsilon) or (self.step_count < self.epsilon_random_steps) or (self.rewards < self.memory_threshold):
            # Epsilon determines the chance of choosing a random/reactive action
            #action_RL = np.random.choice(len(q))
            action = self.RL.random_step()
            self.layer_chosen = 'R'
            #print ('Reactive ', action)
        else:
            # The episodic control policy picks the action with the highest value in QEC for the given state.
            Q_s = self.CL.estimate_return(prototype)
            action = self.CL.action_selection(Q_s)
            self.layer_chosen = 'C'
            #print ('Contextual ', action)

        # Store couplet for next update of STM and LTM based on next reward.
        ac_indx = int(action[0]*self.action_space[0] + action[1])    # convert action[x,x] into an integer for storage
        self.previous_couplet = [prototype, ac_indx]

        return action

    def update_vision(self, obs):
        if self.embbeding_type == 'autoencoder':
            self.PL.advance(obs)
        #print ('Reconstruction Error: ', self.PL.reconstruct_error)

    def update_MFEC(self, last_rwd):
        # PHASE 4. Value update

        disc_return = []
        for t in range((len(self.reward_buffer)-1), -1, -1):
            r_t = 0
            for i in range(len(self.reward_buffer)-t):
                #print("i ", i)
                r_t += np.power(self.discount,i) * self.reward_buffer[t+i]
                #print("np.power(d,i) ", np.power(d,i))
                #print("rwd[t+i] ", rwd[t+i])
            #print("r_t", r_t)
            disc_return.append(r_t)
            self.CL.value_update(self.episode_buffer[t][0],self.episode_buffer[t][1],r_t)
            #if last_rwd > 0: print

        #if last_rwd > 0: print("DISCOUNTED RETURN: ", disc_return)
        if last_rwd > 0: print ("GOAL STATE REACHED! REWARD: ", last_rwd)
        # clean the buffer for the next episode
        self.episode_buffer = []
        self.reward_buffer = []
