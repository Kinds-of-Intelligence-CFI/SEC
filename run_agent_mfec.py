import numpy as np
import random
import platform
from keras import backend as K
import os
import tensorflow as tf

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

    config = tf.ConfigProto()
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

plt = platform.system()
np.seterr(divide = 'ignore')

CodeOcean = False

if CodeOcean == True:
    base_path = '/root/capsule/code/AnimalAI-Olympics/env-win/'
    file_path = '/root/capsule/code/sec/data/simulations/'
    print("OS IS LINUX VM CAPSULE")
else:
    if plt == "Windows":
        base_path = '../AnimalAI-Olympics/env-win/'
        file_path = './data/simulations/'
        set_keras_backend("tensorflow")
        print("OS IS WINDOWS")
    else:
        base_path = '../AnimalAI-Olympics/env-lnx/'
        file_path = './data/simulations/'
        print("OS IS LINUX")

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, MaxPooling2D
from exp_setup_mfec import id_generator, create_env, run_simulation
from agent_mfec import Agent

#################################################################################################################

# SET HYPERPARAMETERS

experiments = 5
episodes = 5000
docker_training = False
environment_visible = False
real_time_plots = False

epsilon = 0.005
discount_factor = 0.99
k_neigbors = 50
action_buffer = 100000  # aka LTM of each action buffer - default 100000
forgetting = "FIFO"
estimation = False
frozen_ws = True

# Number of frames to take random action and observe output
epsilon_random_steps = 1250 # NatureDQN: 50K STEPS / 40 = 1250 steps (5 episodes) animalai
# Number of agent steps for exploration

embedding = 'autoencoder'
prototype_length = 20              #default: 20
reconstruction_threshold = 0.01    #default: 0.005
clr = 4                            #default: 4

#################################################################################################################

# MAIN function
def run_experiment(seed, worker_id):
    print('Testing MFEC Agent with CL_actfreq '+str(clr)+' and LTM_length '+str(action_buffer))

    #seed = random.randint(1,100)
    #worker_id = random.randint(1,10)
    env, arenas = create_env(seed, worker_id, base_path, arenas_n=0, docker=docker_training, env_view=environment_visible, capsule=CodeOcean)

    ID = 'cl'+str(clr)+'-kn'+str(k_neigbors)+'-ltm'+str(action_buffer)+'_agent-'+id_generator(6)+'_' 
    agent = Agent(random_steps=epsilon_random_steps, epsilon=epsilon, discount=discount_factor, k=k_neigbors, ltm_len=action_buffer, embbeding_type=embedding, p_len=prototype_length, rec_thr=reconstruction_threshold, forget=forgetting, estimation=estimation, frozen_ws=frozen_ws, load_ltm=False)
    agent.PL.load_model()

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, cl_r=clr, capsule=CodeOcean)
    print('FINAL SCORE: ', results)
    
    env.close()

#################################################################################################################

# RUN experiment
seed = 0
worker_id = 0

if __name__ == '__main__':
    try:
        for i in range(experiments):
            seed += 1
            worker_id += 1
            print('EXPERIMENT NUMBER ', i)
            run_experiment(seed, worker_id)
    except KeyboardInterrupt:
        print ('Simulation interrumpted!')
