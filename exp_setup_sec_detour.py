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
from arena_generator_lvl4 import create_arena


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
        #arenas = ['3-4-1.yml', '3-4-2.yml']
        arenas = ['3-9-2.yml', '3-7-2.yml', '3-4-1.yml', '3-4-2.yml', '3-4-3.yml', '3-5-1.yml', '3-5-2.yml', '3-5-3.yml', '3-6-1.yml', '3-6-2.yml', '3-6-3.yml', '3-7-1.yml', '3-7-3.yml', '3-8-1.yml', '3-8-2.yml', '3-8-3.yml', '3-9-1.yml', '3-9-3.yml', '3-10-1.yml', '3-10-2.yml', '3-10-3.yml', '3-11-1.yml', '3-11-2.yml', '3-11-3.yml', '3-12-1.yml', '3-12-2.yml', '3-12-3.yml', '4-13-1.yml', '4-13-2.yml', '4-13-3.yml', '4-14-1.yml', '4-14-2.yml', '4-14-3.yml', '4-15-1.yml', '4-15-2.yml', '4-15-3.yml']
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
    trained=False, fp_view=True, dac=True, frameskip=4, stm=20, capsule=True):

    capsule = capsule

    ID = agent_ID
    env = environment
    arenas = arenas_list

    basePath = envPath
    savePath = filePath

    agent = agent_model  
    dac = dac

    action = [0,0] 
    action_space = [3,3]
    action_count = 0

    frameskip = frameskip
    agent_act = True
    agent_view = fp_view

    max_episodes = episodes_n
    episodes = 0

    step_number = 0

    acc_reward = 0              # GET TOTAL REWARD OF BATCH
    rnd_reward = 0              # GET TOTAL REWARD OF THE EPISODE

    log_reward = []             # LOG REWARD PER STEP of the episode
    plot_reward = []            # for visualization
    step_reward = []

    log_mean_rec_err = []       # LOG MEAN RECONSTRUCTION ERROR PER EPISODE
    log_rec_err = []            # LOG RECONSTRUCTION ERROR PER STEP of the episode
    plot_rec_err = []           # for visualization LOG RECONSTRUCTION ERROR PER STEP of the whole batch

    log_speed = []
    log_layer_chosen = []
    log_entropy = []
    log_action = []

    episode_data = []

    reactive_wins = 0
    contextual_wins = 0

    start = time.time()
    elapsed = 0
    ep_memory_full = 0

    summary = {}
    summary['agent_ID'] = ID
    summary['agent_type'] = 'DAC' if dac == True else 'Reactive'  
    summary['total_episodes'] = episodes_n
    summary['total_reward'] = 0 
    
    summary['CL_ratio'] = frameskip
    summary['STM_length'] = stm
    summary['LTM'] = agent.CL.nl if dac == True else False
    summary['decision_inertia'] = agent.CL.decision_inertia if dac == True else False
    summary['forgetting'] = agent.CL.forget if dac == True else False
    summary['action_selection'] = agent.action_selection if dac == True else False
    summary['softmax'] = agent.CL.softmax if dac == True else False
    summary['value_function'] = agent.CL.value_function if dac == True else False

    summary['avg_rec_err'] = 0
    summary['rec_thres'] = agent.reconstruct_thres
    summary['mean_step_rew'] = 0
    summary['mean_step_log'] = 0
    summary['wins_reactive'] = 0
    summary['wins_contextual'] = 0
    summary['auto_reliable'] = 0

    info_dict = env.step(vector_action=[0,0])
    simulation = True


    while simulation:            
        # Set ratio for CL --- STANDARD VALUE: 4
        if step_number%frameskip == 0: agent_act = True
        else: agent_act = False

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

        # STORE REWARDS
        rnd_reward += reward
        log_reward.append(reward)
 
        if (reward >= 1.): # PREVIOUSLY
        #if (rnd_reward >= 0.): 
            print ("N ACTIONS TO REACH REWARD:", action_count)
            print('FINAL SCORE: ', rnd_reward)

            step_reward.append(action_count)
            if dac and len(agent.CL.LTM[2]) > 4: 
                if dac and agent.layer_chosen == 'R': reactive_wins +=1
                if dac and agent.layer_chosen == 'C': contextual_wins +=1
            if dac: agent.CL.updateLTM(rnd_reward) 
            #if dac: agent.CL.updateLTM(reward) # PREVIOUSLY
            if not dac: reactive_wins +=1

        # UPDATE AUTOENCODER AT EVERY STEP
        agent.update_vision(visual_obs) # rec_err is updated here now
        #agent.PL.advance(visual_obs)
        log_speed.append(speed_obs)
        log_rec_err.append(agent.PL.reconstruct_error)

        # TAKE ACTION AT EVERY X STEPS
        if agent_act:
            action_count += 1
            if dac:
                action = agent.step(visual_obs, speed_obs, rnd_reward, agent_done, agent_info)
                #if agent_view: plot_rec_err.append(agent.PL.reconstruct_error)
                log_entropy.append(agent.CL.entropy)
                agent.update_epsilon()
            else:
                action = agent.step(visual_obs, speed_obs, rnd_reward, agent_done, agent_info)
            #if agent_view: plot_reward.append(acc_reward+rnd_reward)
            #print ('Reconstruction Error: ', agent.PL.reconstruct_error)
 
        log_layer_chosen.append(agent.layer_chosen)
        log_action.append(action)

        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        info_dict = env.step(vector_action=action)
        step_number += 1

        # END OF EPISODE
        if agent_done:
            if dac: agent.CL.refresh()       
            acc_reward += rnd_reward
            mean_rec_err = np.mean(log_rec_err)
            log_mean_rec_err.append(mean_rec_err)
            elapsed = time.time() - start
            start = time.time()
            if ep_memory_full == 0 and agent.CL.memory_full == True:
                ep_memory_full = episodes

            if (episodes%1 == 0): 
                print ("EPISODE "+ str(episodes) + " DONE!")
                #print ("TOTAL MEMORIES: "+ str(agent.CL.get_LTM_length()))
                print ("TIME ELAPSED: "+ str(elapsed))
                #print ("FINAL REWARD: "+ str(rnd_reward))
                #print ("N ACTIONS PERFORMED:", action_count)
                #print('Mean Reconstruction Error:', mean_rec_err)

            keys = ['reward_ep', 'rec_error_mean', 'reward_logs', 'rec_error_logs', 'agent_speed_logs', 'active_layer_logs', 'entropy_logs', 'action_sel_logs']
            episode_data = [rnd_reward, mean_rec_err, log_reward, log_rec_err, log_speed, log_layer_chosen, log_entropy, log_action]
            save_data(savePath, ID, episode_data, keys, episodes)

            rnd_reward = 0
            episode_data, log_reward, log_speed, log_rec_err, log_layer_chosen, log_entropy, log_action = [], [], [], [], [], [], []

            action_count = 0
            step_number = 0
            episodes += 1

            if episodes >= max_episodes:
                acc_mean_rec_err = np.mean(log_mean_rec_err)
                mean_step_rew = np.mean(step_reward)
                encoder_reliability = (agent.count_reliable / (agent.count_reliable+agent.count_unreliable)) * 100
                memory_length = agent.CL.get_LTM_length()
                print('FINAL average reconstruction error:', acc_mean_rec_err)
                print('ACTIONS NEEDED TO REACH REWARD:', step_reward)
                print('MEAN ACTIONS NEEDED TO REACH REWARD:', mean_step_rew)
                print('AUTOENCODER RELIABILITY:', encoder_reliability)
                print('REACTIVE REWARDS:', reactive_wins)
                print('CONTEXTUAL REWARDS:', contextual_wins)
                print('TOTAL MEMORIES:', memory_length)
                summary['total_reward'] = acc_reward
                summary['avg_rec_err'] = acc_mean_rec_err
                summary['wins_reactive'] = reactive_wins
                summary['wins_contextual'] = contextual_wins
                summary['auto_reliable'] = encoder_reliability
                summary['mean_step_rew'] = mean_step_rew
                summary['mean_step_log'] = step_reward
                summary['memory_length'] = memory_length
                summary['memory_full'] = ep_memory_full

                save_summary(savePath, ID, summary)
                #agent.PL.save_model(ID)
                #agent.CL.save_LTM(ID)

                #print('FILES SAVED')
                print ('Simulation completed!')
                simulation = False

            if simulation:
                if len(arenas) > 1: 
                    if capsule: 
                        arena_config_in = ArenaConfig('/root/capsule/code/sec/data/utilities/arenas/'+arenas[episodes%len(arenas)])
                    else:
                        arena_config_in = ArenaConfig('./utilities/arenas/'+arenas[episodes%len(arenas)])
                        print("NEW ARENA: ", arenas[episodes%len(arenas)])
                    env.reset(arenas_configurations=arena_config_in, train_mode=True)
                else:
                    env.reset()
                info_dict = env.step(vector_action=[0, 0])

    return acc_reward


def save_data(savePath, ID, data, keys, episodes):
    save = 'w' if episodes == 0 else 'a'
    with open(savePath+ID+'data.csv', save) as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if episodes == 0: writer.writeheader()
        writer.writerow({
            'reward_ep': data[0], 
            'rec_error_mean': data[1], 
            'reward_logs': data[2], 
            'rec_error_logs': data[3], 
            'agent_speed_logs': data[4], 
            'active_layer_logs': data[5], 
            'entropy_logs': data[6], 
            'action_sel_logs': data[7]
        })

def save_summary(savePath, ID, summ):
    with open(savePath+ID+'smry.json', 'w', encoding='utf-8') as f:
        json.dump(summ, f, ensure_ascii=False, indent=4)
