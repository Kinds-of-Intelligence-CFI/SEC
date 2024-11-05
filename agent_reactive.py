import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./DAC/')
from RandomReactiveLayer import ReactiveLayer as RL


class ReactiveAgent(object):

    def __init__(self):
        self.RL = RL()
        self.layer_chosen = 'R' #Reactive Layer default selection

    def reset(self, t=250):
        pass

    def step(self, obs, speed, reward, done, info):
        # ACTION SELECTION PHASE
        # Action proposed by the reactive layer.
        action = self.RL.random_step()

        self.layer_chosen = 'R'
        #print ('Reactive ', action)
        return action
