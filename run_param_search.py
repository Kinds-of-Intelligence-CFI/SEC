import numpy as np
import random
import platform
from tensorflow.python.keras import backend as K
import os, json
import tensorflow as tf
import matplotlib.pyplot as plt

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

    config = tf.compat.v1.ConfigProto()
    jit_level = tf.compat.v1.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)

pltfm = platform.system()

#if os.name == 'nt': # Windows
if pltfm == "Windows":
    base_path = '../AnimalAI-Olympics/'
    file_path = './data/'
    set_keras_backend("tensorflow")
    print("OS IS WINDOWS")
    #file_path = 'D:/SPECS/Projects/AnimalAI-Olympics/autoencoder.h5'
else:
    base_path = '../AnimalAI-Olympics/'
    file_path = './data/'
    print("OS IS NOT WINDOWS")
    #file_path = '/home/simulator/SPECS/AnimalAI-Olympics/animalai/autoencoder.h5'

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input, MaxPooling2D
from exp_setup import id_generator, create_env, run_simulation
from agent_episodic_dac import Agent
from agent_reactive_dac import ReactiveAgent

#################################################################################################################

# SET PARAMETERS

experiments = 20
episodes = 1000
environment_visible = False
real_time_plots = False

dac = True
decision_intertia = True
forgetting = True

#clr = 4
clrs = [4]  #default: 4
#clrs = [4,6,8]  #default: 4

#stm = 20
stms = [50]   #default: 50
#stms = [50,40,30]   #default: 50

#ltm = 25
ltms = [25]   #default: 500
#ltms = [50,40,30]   #default: 500

prototype_length = 20              #default: 20
reconstruction_threshold = 0.01    #default 0.005

#################################################################################################################

# MAIN function
def run_experiment(dac, episodes, clr, stm, ltm):
    print('Testing DAC Agent with CL_actfreq '+str(clr)+' and STM_length '+str(stm)) if dac == True else print('Testing Reactive Agent...')

    seed = random.randint(1,100)
    worker_id = random.randint(1,10)
    env, arenas = create_env(seed, worker_id, base_path, arenas_n=10, env_view=environment_visible)

    ID = 'cl'+str(clr)+'-stm'+str(stm)+'-ltm'+str(ltm)+'_agent-'+id_generator(6)+'_' if dac == True else 'reactive_agent-'+id_generator(6)
    agent = Agent(stm_len=stm, ltm_len=ltm, p_len=prototype_length, rec_thr=reconstruction_threshold, d_ine=decision_intertia, forget=forgetting) if dac == True else ReactiveAgent(p_len=prototype_length, rec_thr=reconstruction_threshold)
    agent.PL.load_model()

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, dac=dac, cl_r=clr, stm=stm)
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

if __name__ == '__main__':
    try:
        for clr in clrs:
            for stm in stms:
                for ltm in ltms: 
                    cond_rewards = []
                    for i in range(experiments):
                        print('STARTING EXPERIMENT CL'+str(clr)+' STM'+str(stm)+' LTM'+str(ltm)+'... NUMBER '+str(i+1))
                        results = run_experiment(dac=dac, episodes=episodes, clr=clr, stm=stm, ltm=ltm)
                        cond_rewards.append(results)

                    cond = 'clr_'+str(clr)+'_stm'+str(stm)
                    psearch_data[cond] = cond_rewards
        
        save_data(file_path, psearch_data, 'rewards')
        save_fig(file_path, psearch_data)

    except KeyboardInterrupt:
        print ('Simulation interrumpted!')
