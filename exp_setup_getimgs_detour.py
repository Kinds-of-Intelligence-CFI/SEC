import sys, os, time, string, random, csv, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./utilities/')
#sys.path.insert(0, "/root/capsule/code/sec/utilities/")

from arena_generator_lvl4 import create_arena

sys.path.append('../AnimalAI-Olympics/animalai/')
#sys.path.insert(0, "/root/capsule/code/AnimalAI-Olympics/animalai/")

from keras.models import Model
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig
from arena_generator_aetrain_detour import create_arena


def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))

def create_env(seed, work_id, basePath, arenas_n=10, docker=True, env_view=True, save_data=False, capsule=True):
    env = UnityEnvironment(
        file_name=basePath+'AnimalAI',  # Path to the environment
        #worker_id=np.random.randint(1,10),  # Unique ID for running the environment (used for connection)
        worker_id=work_id,  # Unique ID for running the environment (used for connection)
        seed=seed,  # The random seed
        docker_training=docker,  # Whether or not you are training inside a docker
        n_arenas=1,  # Number of arenas in your environment
        play=False,  # Set to False for training
        inference=env_view,  # Set to true to watch your agent in action
        resolution=None  # Int: resolution of the agent's square camera (in [4,512], default 84)
    )

    if arenas_n > 0:
        arenas = create_arena(seed, arenas_n)
    else:
        #arenas = ['newdac_01.yaml']
        #arenas = ['3-4-1.yml', '3-4-2.yml', '3-7-2.yml', '3-9-2.yml']
        arenas = ['3-4-1.yml', '3-4-2.yml']
        #arenas = ['3-9-2.yml', '3-7-2.yml', '3-4-1.yml', '3-4-2.yml', '3-4-3.yml', '3-5-1.yml', '3-5-2.yml', '3-5-3.yml', '3-6-1.yml', '3-6-2.yml', '3-6-3.yml', '3-7-1.yml', '3-7-3.yml', '3-8-1.yml', '3-8-2.yml', '3-8-3.yml', '3-9-1.yml', '3-9-3.yml', '3-10-1.yml', '3-10-2.yml', '3-10-3.yml', '3-11-1.yml', '3-11-2.yml', '3-11-3.yml', '3-12-1.yml', '3-12-2.yml', '3-12-3.yml', '4-13-1.yml', '4-13-2.yml', '4-13-3.yml', '4-14-1.yml', '4-14-2.yml', '4-14-3.yml', '4-15-1.yml', '4-15-2.yml', '4-15-3.yml']
        np.random.shuffle(arenas)

    if capsule: 
        arena_config_in = ArenaConfig('/root/capsule/code/sec/data/utilities/arenas/'+arenas[0])
    else:
        arena_config_in = ArenaConfig('./utilities/arenas/'+arenas[0])
    print("GENERATING ENVIRONMENT...")

    env.reset(arenas_configurations=arena_config_in,
              # A new ArenaConfig to use for reset, leave empty to use the last one provided
              train_mode=True  # True for training
              )

    return env, arenas


def run_simulation(agent_ID, agent_model, environment, arenas_list, envPath, filePath, episodes_n=10, 
    trained=False, fp_view=True, save_imgs=True, capsule=True):

    capsule = capsule

    ID = agent_ID
    env = environment
    arenas = arenas_list

    base_path = envPath
    save_path = filePath

    save_img = save_imgs
    img_path = save_path+'env_imgs/'

    agent = agent_model  

    action = [0,0] 
    action_space = [3,3]
    action_count = 0

    agent_act = True
    agent_view = fp_view

    max_episodes = episodes_n
    episodes = 0

    step_number = 0
    start = time.time()
    elapsed = 0

    info_dict = env.step(vector_action=[0,0])
    simulation = True


    while simulation:            
        # GET ENV INFORMATION
        agent_info = info_dict["Learner"]
        #print(agent_info.__dict__.keys())
        visual_obs = agent_info.visual_observations[0]
        #print("visual info", visual_obs.shape)
        #speed_obs = agent_info.vector_observations[0]
        speed_obs = 0
        agent_done = agent_info.local_done[0]
        reward = agent_info.rewards[0]
        #print ("agent info", agent_info)
        #print ("rewards info", agent_info.rewards[0])

        if save_img:
            #print('arena '+arenas[episodes])
            #obs = obs.reshape(1,84,84,3)
            #print('visual_obs img_saved', obs[0].shape)
            plt.imsave(img_path+'png/img_'+str(episodes)+'.png', visual_obs[0])
            np.save(img_path+'npy/img_'+str(episodes), visual_obs[0])
            #print ('visual_obs', visual_obs[0].shape)
            agent_done = True

        action = agent.step(visual_obs, speed_obs, reward, agent_done, agent_info)

        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        info_dict = env.step(vector_action=action)
        step_number += 1

        # END OF EPISODE
        if agent_done:
            #if (episodes%100 == 0):
            print ("EPISODE "+ str(episodes) + " DONE!")
            episodes += 1

            if episodes >= max_episodes:
                print ('Simulation completed!')
                simulation = False

            if simulation:
                if len(arenas) > 1: 
                    if capsule: 
                        arena_config_in = ArenaConfig('/root/capsule/code/sec/data/utilities/arenas/'+arenas[episodes%len(arenas)])
                    else:
                        arena_config_in = ArenaConfig('./utilities/arenas/'+arenas[episodes%len(arenas)])
                        #print("NEW ARENA: ", arenas[episodes%len(arenas)])
                    env.reset(arenas_configurations=arena_config_in, train_mode=True)
                else:
                    env.reset()
                info_dict = env.step(vector_action=[0, 0])

    return reward
