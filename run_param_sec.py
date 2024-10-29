import numpy as np
import random
import platform
from keras import backend as K
import os, json
import tensorflow as tf
import matplotlib.pyplot as plt

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

os_platform = platform.system()
np.seterr(divide = 'ignore')

CodeOcean = False

if CodeOcean == True:
    base_path = '/root/capsule/code/AnimalAI-Olympics/env-win/'
    file_path = '/root/capsule/code/sec/data/simulations/'
    print("OS IS LINUX VM CAPSULE")
else:
    if os_platform == "Windows":
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
from exp_setup_sec import id_generator, create_env, run_simulation
from agent_sec import Agent
from agent_reactive import ReactiveAgent

#################################################################################################################

# SET HYPERPARAMETERS
experiments = 5
episodes = 5000
docker_training = False
environment_visible = False
real_time_plots = False

dac = True

reconstruction_threshold = 0.01    #default 0.005
prototype_length = 20              #default: 20
frozen_ws = True

stms = [50]   #default: 50
#stms = [50,40,30]   #default: 50
ltms = [1000]   #default: 500
#ltms = [50,40,30]   #default: 500
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
def run_experiment(stm, ltm, seed, worker_id):
    print('Testing DAC Agent with CL_actfreq '+str(frameskip)+' and STM_length '+str(stm))

    #seed = random.randint(seed)
    #worker_id = random.randint(worker_id)
    env, arenas = create_env(seed, worker_id, base_path, arenas_n=episodes, docker=docker_training, env_view=environment_visible, capsule=CodeOcean)

    ID = 'SEC_cl'+str(frameskip)+'-stm'+str(stm)+'-ltm'+str(ltm)+'_agent-'+id_generator(6)
    agent = Agent(episodes=episodes, epsilon=max_epsilon, selection=action_selection, rec_thr=reconstruction_threshold,
        p_len=prototype_length, frozen_ws=frozen_ws,
        stm_len=stm, ltm_len=ltm,
        d_ine=decision_inertia, forget=forgetting, load_ltm=load_ltm,
        softmax=softmax, value_function=value_function)

    agent.PL.load_model()

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, dac=dac, frameskip=frameskip, stm=stm, capsule=CodeOcean)
    print('FINAL SCORE: ', results)
    
    env.close()
    return results

#################################################################################################################

# AUX functions
def save_data(savePath, data, name):
    with open(savePath+'param_search_'+name+'.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_fig(savePath, ps_data):
    lists = sorted(ps_data.items()) # sorted by key, return a list of tuples
    keys, data = zip(*lists) # unpack a list of pairs into two tuples

    height = np.mean(data, axis=1)
    y_pos = np.arange(len(keys))
     
    plt.bar(y_pos, height)
    plt.xticks(y_pos, keys)
    plt.savefig(savePath+'param_search_rewards_'+str(seed)+'.png')

#################################################################################################################

# RUN experiment
psearch_data = {}
seed = 0
worker_id = 0

if __name__ == '__main__':
    try:
        for stm in stms:
            for ltm in ltms:
                cond_rewards = []
                for i in range(experiments):
                    seed += 1
                    worker_id += 1
                    print('STARTING EXPERIMENT CL'+str(frameskip)+' STM'+str(stm)+' LTM'+str(ltm)+'... NUMBER '+str(i+1))
                    results = run_experiment(stm=stm, ltm=ltm, seed=seed, worker_id=worker_id)
                    cond_rewards.append(results)

                cond = 'clr_'+str(frameskip)+'_stm'+str(stm)
                psearch_data[cond] = cond_rewards

        save_data(file_path, psearch_data, 'rewards')
        save_fig(file_path, psearch_data)

    except KeyboardInterrupt:
        print ('Simulation interrumpted!')
