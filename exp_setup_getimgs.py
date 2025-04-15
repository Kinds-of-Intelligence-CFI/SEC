import sys, os, time, string, random, csv, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./utilities/')
#sys.path.insert(0, "/root/capsule/code/sec/utilities/")
from arena_generator_detour import create_arena_detour
from arena_generator_doubletmaze import create_arena_doubletmaze
from arena_generator_cylinder import create_arena_cylinder
from arena_generator_permanence import create_arena_permanence
from arena_generator_thorndike import create_arena_thorndike

sys.path.append('../AnimalAI-Olympics/animalai/')
#sys.path.insert(0, "/root/capsule/code/AnimalAI-Olympics/animalai/")

from keras.models import Model
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from animalai.environment import AnimalAIEnvironment, UnityEnvironment
from animalai.actions import AAIActions


def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))

def create_env(seed, work_id, base_path, game_ID, arenas_n=10, docker=True, env_view=True, save_data=False, capsule=True):
    
    timescale = 5
    target_framerate = -1

    if arenas_n > 0:
        if game_ID == 'doubleTmaze': arenas = create_arena_doubletmaze(seed, arenas_n)
        if game_ID == 'detour': arenas = create_arena_detour(seed, arenas_n)
        if game_ID == 'cylinder': arenas = create_arena_cylinder(seed, arenas_n)
        if game_ID == 'permanence': arenas = create_arena_permanence(seed, arenas_n)
        if game_ID == 'thorndike': arenas = create_arena_thorndike(seed, arenas_n)

    else:
        if game_ID == 'doubleTmaze':
            #arenas = ['newdac_01.yaml']
            arenas = ['newdac_01.yaml', 'newdac_02.yaml', 'newdac_03.yaml', 'newdac_04.yaml']
            np.random.shuffle(arenas)
        if game_ID == 'detour':
            arenas = ['3-4-1.yml', '3-4-2.yml']
            #arenas = ['3-4-1.yml', '3-4-2.yml', '3-7-2.yml', '3-9-2.yml']
            #arenas = ['3-9-2.yml', '3-7-2.yml', '3-4-1.yml', '3-4-2.yml', '3-4-3.yml', '3-5-1.yml', '3-5-2.yml', '3-5-3.yml', '3-6-1.yml', '3-6-2.yml', '3-6-3.yml', '3-7-1.yml', '3-7-3.yml', '3-8-1.yml', '3-8-2.yml', '3-8-3.yml', '3-9-1.yml', '3-9-3.yml', '3-10-1.yml', '3-10-2.yml', '3-10-3.yml', '3-11-1.yml', '3-11-2.yml', '3-11-3.yml', '3-12-1.yml', '3-12-2.yml', '3-12-3.yml', '4-13-1.yml', '4-13-2.yml', '4-13-3.yml', '4-14-1.yml', '4-14-2.yml', '4-14-3.yml', '4-15-1.yml', '4-15-2.yml', '4-15-3.yml']
            np.random.shuffle(arenas)
        if game_ID == 'cylinder':
            arenas = ['3-13-1.yml']
            #arenas = ['3-13-1.yml', '3-13-2.yml', '3-13-3.yml']
            #arenas = ['3-15-1.yml', '3-13-1.yml', '3-13-2.yml', '3-13-3.yml', '3-14-1.yml', '3-14-2.yml', '3-14-3.yml', '3-15-2.yml', '3-15-3.yml']
            np.random.shuffle(arenas)
        if game_ID == 'permanence':
            arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml']
            #arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml', '8-10-1.yml','8-10-2.yml', '8-10-3.yml']
            #arenas = ['8-19-2.yml', '8-19-3.yml', '8-20-1.yml', '8-20-3.yml', '8-21-1.yml', '8-21-2.yml', '8-21-3.yml', '8-22-1.yml', '8-22-2.yml', '8-22-3.yml', '8-23-1.yml', '8-23-2.yml', '8-23-3.yml', '8-24-1.yml', '8-24-3.yml', '8-25-1.yml', '8-25-2.yml', '8-25-3.yml', '8-26-1.yml', '8-26-3.yml', '8-27-1.yml', '8-27-2.yml', '8-27-3.yml', '8-28-1.yml', '8-28-2.yml', '8-28-3.yml', '8-29-1.yml', '8-29-2.yml', '8-29-3.yml', '8-30-1.yml', '8-30-2.yml', '8-30-3.yml', '8-7-3.yml', '8-8-1.yml', '8-8-2.yml', '8-8-3.yml', '8-9-1.yml', '8-9-2.yml', '8-9-3.yml', '8-10-1.yml', '8-10-2.yml', '8-10-3.yml', '8-11-1.yml', '8-11-3.yml', '8-12-1.yml', '8-12-2.yml', '8-12-3.yml', '8-13-1.yml', '8-13-2.yml', '8-13-3.yml', '8-14-1.yml', '8-14-2.yml', '8-14-3.yml', '8-15-1.yml', '8-15-2.yml', '8-15-3.yml', '8-16-1.yml', '8-16-2.yml', '8-16-3.yml', '8-17-2.yml', '8-17-3.yml', '8-18-1.yml', '8-18-2.yml', '8-18-3.yml', '8-19-1.yml', '8-24-2.yml', '8-20-2.yml', '8-26-2.yml', '8-11-2.yml', '8-17-1.yml', '8-1-1.yml', '8-1-2.yml', '8-1-3.yml', '8-2-1.yml', '8-2-2.yml', '8-2-3.yml', '8-3-1.yml', '8-3-2.yml', '8-3-3.yml', '8-4-1.yml', '8-4-2.yml', '8-4-3.yml', '8-5-1.yml', '8-5-2.yml', '8-5-3.yml', '8-6-1.yml', '8-6-2.yml', '8-6-3.yml', '8-7-1.yml', '8-7-2.yml']
            np.random.shuffle(arenas)
        if game_ID == 'thorndike':
            arenas = ['3-25-1.yml', '3-25-2.yml', '3-25-3.yml']
            #arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml', '8-10-1.yml','8-10-2.yml', '8-10-3.yml']
            #arenas = ['3-27-3.yml', '3-25-1.yml', '3-25-2.yml', '3-25-3.yml', '3-26-1.yml', '3-26-2.yml', '3-26-3.yml', '3-27-1.yml', '3-27-2.yml', '3-28-1.yml', '3-28-2.yml', '3-28-3.yml', '3-29-1.yml', '3-29-2.yml', '3-29-3.yml', '3-30-1.yml', '3-30-2.yml', '3-30-3.yml']
            np.random.shuffle(arenas)
        if game_ID == 'elimination':
            #arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml']
            #arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml', '8-10-1.yml','8-10-2.yml', '8-10-3.yml']
            arenas = ['5-4-1.yml', '5-4-2.yml', '5-4-3.yml', '5-5-1.yml', '5-5-2.yml', '5-5-3.yml', '5-6-1.yml', '5-6-2.yml', '5-6-3.yml', '5-7-1.yml', '5-7-2.yml', '5-7-3.yml', '5-8-1.yml', '5-8-2.yml', '5-8-3.yml', '5-9-1.yml', '5-9-2.yml', '5-9-3.yml', '5-10-1.yml', '5-10-2.yml', '5-10-3.yml', '5-11-1.yml', '5-11-2.yml', '5-11-3.yml', '5-12-1.yml', '5-12-2.yml', '5-12-3.yml']
            np.random.shuffle(arenas)

    if capsule: 
        arena_config_in = '/root/capsule/code/sec/data/utilities/arenas/'+arenas[0]
    else:
        arena_config_in = './utilities/arenas/'+arenas[0]

    print("GENERATING ENVIRONMENT...")
    print("ARENA: ", arena_config_in)

    aai_env = AnimalAIEnvironment(
        file_name=base_path,  # Path to the environment
        arenas_configurations=arena_config_in,  # need to supply one to start with 
        worker_id=work_id,  # Unique ID for running the environment (used for connection)
        seed=seed,  # The random seed
        play=False,  # Set to False for training
        inference=env_view,  # Set to true to watch your agent in action
        resolution=84,  # Int: resolution of the agent's square camera (in [4,512], default 84)
        timescale=timescale,
        targetFrameRate=target_framerate,
        log_folder="./logs/",
    )
    env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
    aai_env.reset(arenas_configurations=arena_config_in,
              # A new ArenaConfig to use for reset, leave empty to use the last one provided
              )

    return env, arenas


def run_simulation(game_ID, agent_ID, agent_model, environment, arenas_list, envPath, filePath, episodes_n=10,
    trained=False, fp_view=True, save_imgs=True, capsule=True):

    capsule = capsule

    ID = agent_ID
    env = environment
    arenas = arenas_list

    base_path = envPath
    save_path = filePath

    save_img = save_imgs
    img_path = save_path+'env_imgs/'+game_ID+'/'

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

    obs, rew, done, info = env.step(0)
    simulation = True


    while simulation:            

        # need to reset the environemnt if done
        if done:
            obs = env.reset()
        # GET ENV INFORMATION
        agent_info = info
        #print(agent_info.__dict__.keys())
        visual_obs = obs
        #print("visual info", visual_obs.shape)
        speed_obs = 0
        #pos_obs = speed_obs[3:]
        #print ("agent pos", pos_obs)
        #speed_obs = speed_obs[:3]
        #print ("agent speed", speed_obs)
        #speed_obs = 0
        #agent_done = agent_info.local_done[0]
        reward = rew
        #print ("agent info", agent_info)
        #print ("rewards info", agent_info.rewards[0])

        if save_img:
            #print('arena '+arenas[episodes])
            #obs = obs.reshape(1,84,84,3)
            #print('visual_obs img_saved', obs[0].shape)
            os.makedirs(img_path+'png/', exist_ok=True)
            plt.imsave(img_path+'png/img_'+str(episodes)+'.png', visual_obs)
            os.makedirs(img_path+'npy/', exist_ok=True)
            np.save(img_path+'npy/img_'+str(episodes), visual_obs)
            #np.save(img_path+'pos/pos_'+str(episodes), pos_obs)
            #print ('visual_obs', visual_obs[0].shape)
            agent_done = True

        action = agent.step(visual_obs, speed_obs, reward, agent_done, agent_info)

        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        obs, rew, done, info = env.step((action[0]*3)+action[1])
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
                        arena_config_in = '/root/capsule/code/sec/data/utilities/arenas/'+arenas[episodes%len(arenas)]
                    else:
                        arena_config_in = './utilities/arenas/'+arenas[episodes%len(arenas)]
                        print("ARENA: ", arenas[episodes%len(arenas)])
                    env._env.reset(arenas_configurations=arena_config_in)
                else:
                    env.reset()
                # need to reset the environemnt if done
                if done:
                    obs = env.reset()
                obs, rew, done, info = env.step(0)

    return reward
