import numpy as np
import random
import platform
from tensorflow.python.keras import backend as K
import os
import tensorflow as tf

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

plt = platform.system()
np.seterr(divide = 'ignore')

if plt == "Windows":
    base_path = r"C:\AnimalAI\4.2.0\Animal-AI.exe"
    file_path = './data/simulations/'
    set_keras_backend("tensorflow")
    print("OS IS WINDOWS")
else:
    base_path = '../AnimalAI/4.2.0/AAI.x86_64'
    file_path = './data/simulations/'
    print("OS IS LINUX")

from keras.models import Model
from keras.layers import Dense, Flatten, Input, MaxPooling2D
from exp_setup_sec import id_generator, create_env, run_simulation
from agent_reactive import ReactiveAgent
# from agent_adaptive import AdaptiveAgent
from agent_sec import ContextualAgent

#################################################################################################################

# SET HYPERPARAMETERS

game = 'detour'         # envs = ['doubleTmaze', 'detour', 'cylinder', 'permanence', 'thorndike', 'elimination']
experiments = 5
episodes = 5000
environment_visible = False
real_time_plots = False

dac = True
decision_intertia = True
forgetting = "NONE"
# forgetting = ['NONE', 'FIFO', 'RWD', 'RWD-PROP', 'PRIOR']
value_function = 'default'
# value_functions = ['default', 'noGi', 'noDist', 'noRR', 'soloGi', 'soloDist', 'soloRR']

stm = 50
#stms = [40,60,80,100]  #default: 40
ltm = 500
#stms = [25,50,100,500] #default: 500
clr = 4
#clrs = [1,2,4]     #default: 4

prototype_length = 20              #default: 20
reconstruction_threshold = 0.01    #default 0.005

#################################################################################################################

# MAIN function
def run_experiment(dac, episodes, clr, stm, env, arenas):
    print('Testing DAC Agent with CL_actfreq '+str(clr)+' and STM_length '+str(stm)) if dac == True else print('Testing Reactive Agent...')
    #reward = test_agent(seed, worker_id, base_path, dac=dac, episodes=episodes, clr=clr, stm=stm)

    

    ID = 'SEC_'+str(game)+'_cl'+str(clr)+'-stm'+str(stm)+'-ltm'+str(ltm)+'_agent-'+id_generator(6)+'_' if dac == True else 'reactive_agent-'+id_generator(6)
    agent = ContextualAgent(stm_len=stm, ltm_len=ltm, p_len=prototype_length, rec_thr=reconstruction_threshold, d_ine=decision_intertia, forget=forgetting, value_function=value_function) if dac == True else ReactiveAgent()
    if dac: agent.PL.load_saved_model(gameID=game)

    results = run_simulation(ID, agent, env, arenas, base_path, file_path, episodes_n=episodes, fp_view=real_time_plots, dac=dac, cl_r=clr, stm=stm)
    print('FINAL SCORE: ', results)

    env.close()

#################################################################################################################

worker_id = 5

# RUN experiment
if __name__ == '__main__':
    try:
        for i in range(experiments):
            print('EXPERIMENT NUMBER ', i)
            
            seed = random.randint(1,100)
            #worker_id = random.randint(1,10)
            worker_id = random.randint(1,10)
            env, arenas = create_env(seed, worker_id, base_path, game, arenas_n=10, env_view=environment_visible)
            run_experiment(dac=dac, episodes=episodes, clr=clr, stm=stm, env=env, arenas=arenas)
            worker_id += 1
    except KeyboardInterrupt:
        print ('Simulation interrumpted!')
    finally:
        env.close()
