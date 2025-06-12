import sys, os, time, string, random, csv, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./utilities/')
from arena_generator_lvl4 import create_arena

sys.path.append('../AnimalAI-Olympics/animalai/')

from matplotlib.animation import FuncAnimation
from keras.models import Model
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from animalai.environment import AnimalAIEnvironment, UnityEnvironment
from animalai.actions import AAIActions


def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))

def create_env(seed, work_id, base_path, game_ID='doubleTmaze', arenas_n=10, env_view=True, save_data=False):
    timescale = 30
    target_framerate = -1
    print(f"base_path: {base_path}")
    

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

    arena_config_in = './utilities/arenas/'+arenas[0]

    print("GENERATING ENVIRONMENT...")
    print("ARENA: ", arena_config_in)

    port = 5005 + random.randint(0, 1000)
    aai_env = AnimalAIEnvironment(
        file_name=base_path,  # Path to the environment
        arenas_configurations=arena_config_in,  # need to supply one to start with 
        seed=seed,  # The random seed
        play=False,  # Set to False for training
        inference=False,  # Set to true to watch your agent in action
        resolution=84,  # Int: resolution of the agent's square camera (in [4,512], default 84)
        useRayCasts=False,
        useCamera=True,
        timescale=timescale,
        targetFrameRate=target_framerate,
        base_port=port,
    )
    env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
    aai_env.reset(arenas_configurations=arena_config_in,
              # A new ArenaConfig to use for reset, leave empty to use the last one provided
              )

    return env, arenas


def run_simulation(agent_ID, agent_model, environment, arenas_list, envPath, filePath, episodes_n=10, trained=False, fp_view=True, dac=True, cl_r=4, stm=20):

    ID = agent_ID
    env = environment
    arenas = arenas_list
    episodes = 0
    max_episodes = episodes_n
    basePath = envPath
    savePath = filePath

    agent = agent_model  
    dac = dac
    action = [0,0] 
    CL_act = True
    FIG_view = True   
    CL_ratio = cl_r
    STM_length = stm
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

    summary = {}
    summary['agent_ID'] = ID
    summary['agent_type'] = 'DAC' if dac == True else 'Reactive'
    summary['total_episodes'] = episodes_n
    summary['total_reward'] = 0 
    summary['avg_rec_err'] = 0
    summary['CL_ratio'] = CL_ratio
    summary['STM_length'] = STM_length
    summary['rec_thres'] = agent.reconstruct_thres if dac == True else 0
    summary['mean_step_rew'] = 0
    summary['mean_step_log'] = 0
    summary['wins_reactive'] = 0
    summary['wins_contextual'] = 0
    summary['auto_reliable'] = 0

    obs, rew, done, info = env.step(0)
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

        if optionB:
            plt.ion()
            plt.style.use('seaborn')

            fig1, ax1 = plt.subplots()
            plt.show()


    while simulation:            
        # Set ratio for CL --- STANDARD VALUE: 4
        if step_number%CL_ratio == 0: CL_act = True
        else: CL_act = False

        if step_number%(CL_ratio*5) == 0: FIG_view = True
        else: FIG_view = False

        # need to reset the environemnt if done
        if done:
            obs = env.reset()

        # GET ENV INFORMATION
        #print(f"info dict: {info_dict}")
        agent_info = info
        #print(agent_info.__dict__.keys())
        visual_obs = obs
        speed_obs = 0
        #print ("agent speed", speed_obs)
        #pos = speed_obs[3:]
        #print ("agent pos", pos)
        #speed_obs = speed_obs[:3]
        #print ("agent speed2", speed_obs)
        agent_done = done
        reward = rew
        penalty += rew
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
            if dac: agent.CL.updateLTM(rnd_reward) # ESTA ES LA BUENA
            #if dac: agent.CL.updateLTM(reward) # PREVIOUSLY
            if not dac: reactive_wins +=1

        # UPDATE AUTOENCODER AT EVERY STEP
        if dac: agent.update_vision(visual_obs) # rec_err is updated here now
        #agent.PL.advance(visual_obs)
        log_speed.append(speed_obs)
        if dac: log_rec_err.append(agent.PL.reconstruct_error)

        # TAKE ACTION AT EVERY X STEPS
        if CL_act:
            action_count += 1
            if dac:
                action = agent.step(visual_obs, speed_obs, reward, agent_done, agent_info)
                #if agent_view: plot_rec_err.append(agent.PL.reconstruct_error)
                log_entropy.append(agent.CL.entropy)
            else:
                action = agent.step(visual_obs, speed_obs, reward, agent_done, agent_info)
            #if agent_view: plot_reward.append(acc_reward+rnd_reward)
            #print ('Reconstruction Error: ', agent.PL.reconstruct_error)
 
        log_layer_chosen.append(agent.layer_chosen)
        log_action.append(action)
        #action = [0, 0] # WHY IS THIS HERE?????

        
        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        obs, rew, done, info = env.step((action[0]*3)+action[1])
        step_number += 1

        # END OF EPISODE
        if agent_done:
            if dac: agent.CL.refresh()       
            acc_reward += rnd_reward
            mean_rec_err = np.mean(log_rec_err)
            log_mean_rec_err.append(mean_rec_err)
            elapsed = time.time() - start
            start = time.time()

            if (episodes%1 == 0): 
                print ("EPISODE "+ str(episodes) + " DONE!")
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
                encoder_reliability = (agent.count_reliable / (agent.count_reliable+agent.count_unreliable)) * 100
                print('FINAL average reconstruction error:', acc_mean_rec_err)
                print('ACTIONS NEEDED TO REACH REWARD:', step_reward)
                print('MEAN ACTIONS NEEDED TO REACH REWARD:', mean_step_rew)
                print('AUTOENCODER RELIABILITY:', encoder_reliability)
                print('REACTIVE REWARDS:', reactive_wins)
                print('CONTEXTUAL REWARDS:', contextual_wins)
                summary['total_reward'] = acc_reward
                summary['avg_rec_err'] = acc_mean_rec_err
                summary['wins_reactive'] = reactive_wins
                summary['wins_contextual'] = contextual_wins
                summary['auto_reliable'] = encoder_reliability
                summary['mean_step_rew'] = mean_step_rew
                summary['mean_step_log'] = step_reward

                save_summary(savePath, ID, summary)
                #agent.PL.save_model(ID)
                #agent.CL.save_LTM(ID)

                #print('FILES SAVED')
                print ('Simulation completed!')
                simulation = False

            if simulation:
                if len(arenas) > 1: 
                    arena_config_in = './utilities/arenas/'+arenas[episodes%len(arenas)]
                    #arena_config_in = ArenaConfig(basePath+'examples/configs/'+arenas[episodes%len(arenas)])
                    env._env.reset(arenas_configurations=arena_config_in)
                    #print('ARENA: ', arenas[episodes%len(arenas)])
                else:
                    env.reset()
                # need to reset the environemnt if done
                if done:
                    obs = env.reset()
                obs, rew, done, info = env.step(0)

        if agent_view and FIG_view and dac :    
            if optionA:
                axim1a.set_data(visual_obs[0])
                axim1b.set_data(agent.PL.get_reconstructed_img(visual_obs))
                fig1.canvas.flush_events()

        if agent_view and CL_act and dac and len(agent.CL.LTM[2]) > 1:
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
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
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
