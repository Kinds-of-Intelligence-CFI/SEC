import sys, os, time, string, random, csv, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./utilities/')
#sys.path.insert(0, "/root/capsule/code/sec/utilities/")
from arena_generator_lvl4 import create_arena

sys.path.append('../AnimalAI-Olympics/animalai/')
#sys.path.insert(0, "/root/capsule/code/AnimalAI-Olympics/animalai/")

from matplotlib.animation import FuncAnimation
from keras.models import Model
from animalai.envs import UnityEnvironment
from animalai.envs.arena_config import ArenaConfig


def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))

def create_env(seed, work_id, basePath, game_ID='doubleTmaze', arenas_n=10, docker=True, env_view=True, save_data=False, capsule=True):
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
            #arenas = ['5-4-1.yml', '5-4-2.yml', '5-4-3.yml']
            #arenas = ['5-4-1.yml', '5-4-2.yml', '5-4-3.yml', '5-5-1.yml', '5-5-2.yml', '5-5-3.yml']
            arenas = ['5-4-1.yml', '5-4-2.yml', '5-4-3.yml', '5-5-1.yml', '5-5-2.yml', '5-5-3.yml', '5-6-1.yml', '5-6-2.yml', '5-6-3.yml', '5-7-1.yml', '5-7-2.yml', '5-7-3.yml', '5-8-1.yml', '5-8-2.yml', '5-8-3.yml', '5-9-1.yml', '5-9-2.yml', '5-9-3.yml', '5-10-1.yml', '5-10-2.yml', '5-10-3.yml', '5-11-1.yml', '5-11-2.yml', '5-11-3.yml', '5-12-1.yml', '5-12-2.yml', '5-12-3.yml']
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
    trained=False, fp_view=True, cl_r=4, capsule=True):

    ID = agent_ID
    env = environment
    arenas = arenas_list
    episodes = 0
    max_episodes = episodes_n
    basePath = envPath
    savePath = filePath

    agent = agent_model  
    action = [0,0] 
    CL_act = True
    FIG_view = True   
    CL_ratio = cl_r
    agent_view = fp_view
    penalty = 0
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

    step_number = 0
    action_count = 0
    start = time.time()
    elapsed = 0
    ep_memory_full = 0

    summary = {}
    summary['agent_ID'] = ID
    summary['agent_type'] = 'MFEC'  
    summary['total_episodes'] = episodes_n
    summary['total_reward'] = 0 
    
    summary['CL_ratio'] = CL_ratio
    summary['LTM'] = agent.CL.nl 
    summary['epsilon'] = agent.epsilon
    summary['discount'] = agent.CL.discount
    summary['K_neighbors'] = agent.CL.k
    summary['forgetting'] = agent.CL.forget
    summary['estimation'] = agent.CL.estimation
    #summary['frozen_ws'] = agent.PL.frozen

    summary['avg_rec_err'] = 0
    summary['rec_thres'] = agent.reconstruct_thres
    summary['mean_step_rew'] = 0
    summary['mean_step_log'] = 0
    summary['wins_reactive'] = 0
    summary['wins_contextual'] = 0
    summary['auto_reliable'] = 0

    info_dict = env.step(vector_action=[0,0])
    simulation = True

    optionA = True
    optionB = False
    optionC = False

    if agent_view:
        if optionA: 
            IMAGE_SIZE = 500
            plt.ion()
            plt.style.use('seaborn')

            fig1, ax1 = plt.subplots(1, 2)
            fig2, ax2 = plt.subplots()
            fig3, ax3 = plt.subplots(1, 2)

            ax1[0].set_title("Visual Input")
            ax1[0].grid(False)
            ax1[0].get_yaxis().set_visible(False)
            ax1[0].get_xaxis().set_visible(False)

            ax1[1].set_title("Reconstructed Image")
            ax1[1].grid(False)
            ax1[1].get_yaxis().set_visible(False)
            ax1[1].get_xaxis().set_visible(False)

            ax2.set_title("Contextual layer")
            ax2.grid(False)               

            plt.xlim([0,agent.CL.ns])
            plt.xticks(np.linspace(0, agent.CL.ns, 6, endpoint=True))
            
            # this example doesn't work because array only contains zeroes
            array = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            axim1a = ax1[0].imshow(array)
            axim1b = ax1[1].imshow(array)

            array = np.zeros(shape=(agent.CL.nl, agent.CL.ns), dtype=np.uint8)
            array[0, 0] = 99 # this value allow imshow to initialise it's color scale
            axim2 = ax2.imshow(array, interpolation='none', cmap='copper', aspect='auto')

            del array

            ax3[0].set_title("Reward")
            ax3[1].set_title("Reconstruciton error")


        if optionB:
            plt.ion()
            plt.style.use('seaborn')

            fig1, ax1 = plt.subplots()
            plt.show()

    print("STARTING SIMULATION...")
    while simulation:            
        # Set ratio for CL --- STANDARD VALUE: 4
        if step_number%CL_ratio == 0: CL_act = True
        else: CL_act = False

        if step_number%(CL_ratio*5) == 0: FIG_view = True
        else: FIG_view = False

        # GET ENV INFORMATION
        agent_info = info_dict["Learner"]
        #print(agent_info.__dict__.keys())
        visual_obs = agent_info.visual_observations[0]
        #speed_obs = agent_info.vector_observations[0]
        speed_obs = 0
        agent_done = agent_info.local_done[0]
        reward = agent_info.rewards[0]
        penalty += agent_info.rewards[0]
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

            if agent.layer_chosen == 'R': reactive_wins +=1
            if agent.layer_chosen == 'C': contextual_wins +=1
            
            agent.rewards += 1
            # agent.CL.updateLTM(rnd_reward) 
            #if dac: agent.CL.updateLTM(reward) # PREVIOUSLY

        # UPDATE AUTOENCODER AT EVERY STEP
        agent.update_vision(visual_obs) # rec_err is updated here now
        #agent.PL.advance(visual_obs)
        #log_speed.append(speed_obs)
        log_rec_err.append(agent.PL.reconstruct_error)

        # TAKE ACTION AT EVERY X STEPS
        if CL_act:
            action_count += 1
            rwd = rnd_reward if (reward > 0.) else reward
            action = agent.step(visual_obs, speed_obs, rwd, agent_done, agent_info)
            #if agent_view: plot_rec_err.append(agent.PL.reconstruct_error)
            log_entropy.append(agent.CL.entropy)

            #if agent_view: plot_reward.append(acc_reward+rnd_reward)
            #print ('Reconstruction Error: ', agent.PL.reconstruct_error)
 
        log_layer_chosen.append(agent.layer_chosen)
        log_action.append(action)

        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        info_dict = env.step(vector_action=action)
        step_number += 1

        # END OF EPISODE
        if agent_done:      
            agent.update_MFEC(rnd_reward)
            acc_reward += rnd_reward
            mean_rec_err = np.mean(log_rec_err)
            log_mean_rec_err.append(mean_rec_err)
            elapsed = time.time() - start
            start = time.time()
            if ep_memory_full == 0 and agent.CL.memory_full == True:
                ep_memory_full = episodes

            if (episodes%1 == 0): 
                print ("EPISODE "+ str(episodes) + " DONE!")
                print ("TOTAL MEMORIES: "+ str(agent.CL.get_LTM_length()))
                print ("TIME ELAPSED: "+ str(elapsed))

                #print ("FINAL REWARD: "+ str(rnd_reward))
                #print ("N ACTIONS PERFORMED:", action_count)
                #print('Mean Reconstruction Error:', mean_rec_err)

            keys = ['reward_ep', 'rec_error_mean', 'reward_logs', 'rec_error_logs', 'agent_speed_logs', 'active_layer_logs', 'entropy_logs', 'action_sel_logs']
            episode_data = [rnd_reward, mean_rec_err, log_reward, log_rec_err, log_speed, log_layer_chosen, log_entropy, log_action]
            save_data(savePath, ID, episode_data, keys, episodes)

            rnd_reward = 0
            penalty = 0
            episode_data, log_reward, log_speed, log_rec_err, log_layer_chosen, log_entropy, log_action = [], [], [], [], [], [], []

            action_count = 0
            step_number = 0
            episodes += 1

            if episodes >= max_episodes:
                acc_mean_rec_err = np.mean(log_mean_rec_err)
                mean_step_rew = np.mean(step_reward)
                #encoder_reliability = (agent.count_reliable / (agent.count_reliable+agent.count_unreliable)) * 100
                memory_length = agent.CL.get_LTM_length()
                print('FINAL average reconstruction error:', acc_mean_rec_err)
                print('ACTIONS NEEDED TO REACH REWARD:', step_reward)
                print('MEAN ACTIONS NEEDED TO REACH REWARD:', mean_step_rew)
                #print('AUTOENCODER RELIABILITY:', encoder_reliability)
                print('REACTIVE REWARDS:', reactive_wins)
                print('CONTEXTUAL REWARDS:', contextual_wins)
                print('TOTAL MEMORIES:', memory_length)
                summary['total_reward'] = acc_reward
                summary['avg_rec_err'] = acc_mean_rec_err
                summary['wins_reactive'] = reactive_wins
                summary['wins_contextual'] = contextual_wins
                #summary['auto_reliable'] = encoder_reliability
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
                    env.reset(arenas_configurations=arena_config_in, train_mode=True)
                else:
                    env.reset()
                info_dict = env.step(vector_action=[0, 0])

        if agent_view and FIG_view:
            if optionA:
                axim1a.set_data(visual_obs[0])
                axim1b.set_data(agent.PL.get_reconstructed_img(visual_obs))
                fig1.canvas.flush_events()

                ax3[0].set_title("Reward")
                ax3[1].set_title("Reconstruciton error")     
                ax3[0].cla()
                ax3[0].grid(False)
                ax3[0].set_title("Reward")
                ax3[0].plot(log_reward, color='green')
               
                ax3[1].cla()
                ax3[1].grid(False)
                ax3[1].set_title("Reconstruction Error")
                #ax3[1].set_ylim(0, 100)
                ax3[1].plot(log_rec_err, color='red')
                fig3.canvas.flush_events()

                #plt.draw()
                #plt.pause(0.0001)

        if agent_view and CL_act and len(agent.CL.LTM[2]) > 1:
            if optionA:
                array1 = np.array(agent.CL.selected_actions_indx)
                array2 = np.array(array1) * 255
                axim2.set_data(array2)
                fig2.canvas.flush_events()

            if optionB:
                plt.locator_params(axis='y', nbins=5)

                ax1.set_title("Contextual layer")
                ax1.grid(False)
                ax1.get_yaxis().set_visible(False)
                ax1.get_xaxis().set_visible(False)
                ax1.imshow(agent.CL.selected_actions_indx, interpolation='none', cmap='copper', aspect='auto')

                plt.draw()
                plt.pause(0.0001)

            if optionC:
                plt.locator_params(axis='y', nbins=5)

                A = np.copy(agent.CL.selected_actions_indx)
                X,Y = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))

                ax1.set_title("Contextual layer")
                ax1.grid(False)
                ax1.get_yaxis().set_visible(False)
                ax1.get_xaxis().set_visible(False)
                ax1.scatter(X.flatten(), Y.flatten(), c=A.flatten())

                plt.draw()
                plt.pause(0.0001)

    return acc_reward


def plot_twist(obs, rec_img, pred_img, rew, rec_err, pre_err):
    fig, ax = plt.subplots(2,3)

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
