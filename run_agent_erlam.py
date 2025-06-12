import numpy as np
import random
import platform
from keras import backend as K
import os
import tensorflow as tf

plt = platform.system()
np.seterr(divide = 'ignore')
CodeOcean = False

if CodeOcean == True:
    base_path = '/root/capsule/code/AnimalAI-Olympics/env-win/'
    file_path = '/root/capsule/code/sec/data/simulations/'
    print("OS IS LINUX VM CAPSULE")
else:
    if plt == "Windows":
        base_path = r"C:\AnimalAI\4.2.0\Animal-AI.exe"
        file_path = './data/simulations/'
        print("OS IS WINDOWS")
    else:
        base_path = '../AnimalAI/4.2.0/AAI.x86_64'
        file_path = './data/simulations/'
        print("OS IS LINUX")

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, MaxPooling2D
from exp_setup_erlam import id_generator, create_env, run_simulation
from agent_erlam import Agent

#################################################################################################################

# SET HYPERPARAMETERS

game = 'detour'         # envs = ['doubleTmaze', 'detour', 'cylinder', 'permanence']
experiments = 5
episodes = 5000
docker_training = False
environment_visible = False
real_time_plots = False

# Nat DQN params
learning_rate = 0.00025 #0.25e-3
epsilon_max = 1
epsilon_min = 0.1
#epsilon_eval = 0.05
#epsilon_decay = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken #~0.9994

# Number of frames to take random action and observe output
epsilon_random_steps = 1250 # NatureDQN: 50K STEPS / 40 = 1250 steps (5 episodes) animalai
# Number of agent steps for exploration
epsilon_greedy_steps = 25000 # NatureDQN: 1M STEPS / 40 = 25000 steps (100 episodes) animalai
# Working before with 5000 steps (20 episodes) animalai

# DQN params
gamma = 0.99
lambda_ = 0.3
dqn_update_freq = 4 # update every 4 actions
target_update_freq = 2500  #NatureDQN: 10k steps - 10 episodes

# Memory params
assoc_freq = 50 # 50 epsiodes
batch_size = 32 # 32 experiences
memory_buffer = 25000  # #NatureDQN: 1M/100K - TEST BETWEEN 2500 AND 25000
forgetting = True  # default: True
threshold = 0.02   # DEFAULT SEC AE dim 20
# Autoencoder params
prototype_length = 20              #default: 20
reconstruction_threshold = 0.01    #default: 0.005

frameskip = 4                            #default: 4
input_size = [84, 84, 3]
output_size = 9

#################################################################################################################

# MAIN function
def run_experiment(seed, worker_id):
    print('Testing ERLAM Agent...')

    #seed = random.randint(1,100)
    #worker_id = random.randint(1,10)
    env, arenas = create_env(seed, worker_id, base_path, game, arenas_n=10, docker=docker_training, env_view=environment_visible, capsule=CodeOcean)

    ID = 'ERLAM_'+str(game)+'_cl'+str(frameskip)+'-k'+str(assoc_freq)+'-ltm'+str(memory_buffer)+'_agent-'+id_generator(6)+'_'

    agent = Agent(gameID=game, input_size=input_size, output_size=output_size,
        random_steps=epsilon_random_steps, egreedy_steps=epsilon_greedy_steps, eps_max=epsilon_max, eps_min=epsilon_min,
        lr=learning_rate, gamma=gamma, lamb=lambda_,
        batch_size=batch_size, mem_size=memory_buffer, forgetting=forgetting,
        dqn_update_freq=dqn_update_freq, target_update_freq=target_update_freq, assoc_freq=assoc_freq, threshold=threshold)

    #agent.PL.load_model()

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, frameskip=frameskip, capsule=CodeOcean)
    print('FINAL SCORE: ', results)
    
    env.close()

#################################################################################################################

# RUN experiment
seed = 0
worker_id = 5

if __name__ == '__main__':
    try:
        for i in range(experiments):
            seed += 1
            worker_id += 1
            print('EXPERIMENT NUMBER ', i)
            run_experiment(seed, worker_id)
    except KeyboardInterrupt:
        print ('Simulation interrumpted!')

