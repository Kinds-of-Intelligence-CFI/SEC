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
from exp_setup_sec_detour import id_generator, create_env, run_simulation
from agent_sec import Agent
from agent_reactive import ReactiveAgent

#################################################################################################################

# SET HYPERPARAMETERS

experiments = 5
episodes = 5000
docker_training = False
environment_visible = True
real_time_plots = False

dac = True

reconstruction_threshold = 0.01    #default 0.005
prototype_length = 20              #default: 20
frozen_ws = True

stm = 50    #stms = [40,60,80,100]  #default: 40
ltm = 500   #ltms = [25,50,100,500] #default: 500
decision_inertia = True
forgetting = "FIFO"
#forgettings = ["NONE","FIFO","SING,"PROP"]   #default: FIFO
load_ltm = False

max_epsilon = 1
action_selection = 'default'
# action_selection = ['default', 'e-greedy']
softmax = False
value_function = 'default'
# value_functions = ['default', 'noGi', 'noDist', 'noRR', 'soloGi', 'soloDist', 'soloRR']

frameskip = 4
#frameskip = [1,2,4]     #default: 4


#################################################################################################################

# MAIN function
def run_experiment(seed, worker_id):
    print('Testing DAC Agent with CL_actfreq '+str(frameskip)+' and STM_length '+str(stm))

    #seed = random.randint(1,100)
    #worker_id = random.randint(1,10)
    env, arenas = create_env(seed, worker_id, base_path, arenas_n=0, docker=docker_training, env_view=environment_visible, capsule=CodeOcean)

    ID = 'SEC_cl'+str(frameskip)+'-stm'+str(stm)+'-ltm'+str(ltm)+'_agent-'+id_generator(6)+'_'

    agent = Agent(episodes=episodes, epsilon=max_epsilon, selection=action_selection, rec_thr=reconstruction_threshold,
        p_len=prototype_length, frozen_ws=frozen_ws,
        stm_len=stm, ltm_len=ltm,
        d_ine=decision_inertia, forget=forgetting, load_ltm=load_ltm,
        softmax=softmax, value_function=value_function)

    agent.PL.load_model()

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, dac=dac, frameskip=frameskip, stm=stm, capsule=CodeOcean)
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
