import sys, os, time, string, random, csv, json
import numpy as np

sys.path.append('./utilities/')
#sys.path.insert(0, "/root/capsule/code/sec/utilities/")
from arena_generator_lvl4 import create_arena

sys.path.append('../AnimalAI-Olympics/animalai/')
#sys.path.insert(0, "/root/capsule/code/AnimalAI-Olympics/animalai/")

from keras.models import Model
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from animalai.environment import AnimalAIEnvironment, UnityEnvironment
from animalai.actions import AAIActions

def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for i in range(length))

def create_env(seed, work_id, basePath, gameID='doubleTmaze', arenas_n=10, docker=True, env_view=True, save_data=False, capsule=True):
    
    timescale = 30
    target_framerate = -1

    if arenas_n > 0:
        arenas = create_arena(seed, arenas_n)
    else:
        if gameID == 'doubleTmaze':
            #arenas = ['newdac_01.yaml']
            arenas = ['newdac_01.yaml', 'newdac_02.yaml', 'newdac_03.yaml', 'newdac_04.yaml']
            np.random.shuffle(arenas)
        if gameID == 'detour':
            arenas = ['3-4-1.yml', '3-4-2.yml']
            #arenas = ['3-4-1.yml', '3-4-2.yml', '3-7-2.yml', '3-9-2.yml']
            #arenas = ['3-9-2.yml', '3-7-2.yml', '3-4-1.yml', '3-4-2.yml', '3-4-3.yml', '3-5-1.yml', '3-5-2.yml', '3-5-3.yml', '3-6-1.yml', '3-6-2.yml', '3-6-3.yml', '3-7-1.yml', '3-7-3.yml', '3-8-1.yml', '3-8-2.yml', '3-8-3.yml', '3-9-1.yml', '3-9-3.yml', '3-10-1.yml', '3-10-2.yml', '3-10-3.yml', '3-11-1.yml', '3-11-2.yml', '3-11-3.yml', '3-12-1.yml', '3-12-2.yml', '3-12-3.yml', '4-13-1.yml', '4-13-2.yml', '4-13-3.yml', '4-14-1.yml', '4-14-2.yml', '4-14-3.yml', '4-15-1.yml', '4-15-2.yml', '4-15-3.yml']
            np.random.shuffle(arenas)
        if gameID == 'cylinder':
            arenas = ['3-13-1.yml']
            #arenas = ['3-13-1.yml', '3-13-2.yml', '3-13-3.yml']
            #arenas = ['3-15-1.yml', '3-13-1.yml', '3-13-2.yml', '3-13-3.yml', '3-14-1.yml', '3-14-2.yml', '3-14-3.yml', '3-15-2.yml', '3-15-3.yml']
            np.random.shuffle(arenas)
        if gameID == 'permanence':
            arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml']
            #arenas = ['8-7-1.yml', '8-7-2.yml', '8-7-3.yml', '8-10-1.yml','8-10-2.yml', '8-10-3.yml']
            #arenas = ['8-19-2.yml', '8-19-3.yml', '8-20-1.yml', '8-20-3.yml', '8-21-1.yml', '8-21-2.yml', '8-21-3.yml', '8-22-1.yml', '8-22-2.yml', '8-22-3.yml', '8-23-1.yml', '8-23-2.yml', '8-23-3.yml', '8-24-1.yml', '8-24-3.yml', '8-25-1.yml', '8-25-2.yml', '8-25-3.yml', '8-26-1.yml', '8-26-3.yml', '8-27-1.yml', '8-27-2.yml', '8-27-3.yml', '8-28-1.yml', '8-28-2.yml', '8-28-3.yml', '8-29-1.yml', '8-29-2.yml', '8-29-3.yml', '8-30-1.yml', '8-30-2.yml', '8-30-3.yml', '8-7-3.yml', '8-8-1.yml', '8-8-2.yml', '8-8-3.yml', '8-9-1.yml', '8-9-2.yml', '8-9-3.yml', '8-10-1.yml', '8-10-2.yml', '8-10-3.yml', '8-11-1.yml', '8-11-3.yml', '8-12-1.yml', '8-12-2.yml', '8-12-3.yml', '8-13-1.yml', '8-13-2.yml', '8-13-3.yml', '8-14-1.yml', '8-14-2.yml', '8-14-3.yml', '8-15-1.yml', '8-15-2.yml', '8-15-3.yml', '8-16-1.yml', '8-16-2.yml', '8-16-3.yml', '8-17-2.yml', '8-17-3.yml', '8-18-1.yml', '8-18-2.yml', '8-18-3.yml', '8-19-1.yml', '8-24-2.yml', '8-20-2.yml', '8-26-2.yml', '8-11-2.yml', '8-17-1.yml', '8-1-1.yml', '8-1-2.yml', '8-1-3.yml', '8-2-1.yml', '8-2-2.yml', '8-2-3.yml', '8-3-1.yml', '8-3-2.yml', '8-3-3.yml', '8-4-1.yml', '8-4-2.yml', '8-4-3.yml', '8-5-1.yml', '8-5-2.yml', '8-5-3.yml', '8-6-1.yml', '8-6-2.yml', '8-6-3.yml', '8-7-1.yml', '8-7-2.yml']
            np.random.shuffle(arenas)

    if capsule: 
        arena_config_in = '/root/capsule/code/sec/data/utilities/arenas/'+arenas[0]
    else:
        arena_config_in = './utilities/arenas/'+arenas[0]
    print("GENERATING ENVIRONMENT...")
    print("ARENA: ", arena_config_in)

    port = 5005 + random.randint(0, 1000)
    aai_env = AnimalAIEnvironment(
        file_name=basePath,  # Path to the environment
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


def run_simulation(agent_ID, agent_model, environment, arenas_list, envPath, filePath, episodes_n=10, 
    trained=False, fp_view=True, frameskip=4, capsule=True):

    ID = agent_ID
    env = environment
    arenas = arenas_list

    basePath = envPath
    savePath = filePath

    agent = agent_model  

    action = [0,0] 
    action_space = [3,3]

    frameskip = frameskip
    frame_count = 0

    agent_steps = 0
    agent_act = True
    agent_view = fp_view

    episode_count = 0
    episode_frames = 0
    max_episodes = episodes_n

    total_reward = 0              # GET TOTAL REWARD OF EXPERIMENT
    episode_reward = 0              # GET TOTAL REWARD OF THE EPISODE

    eps_memory_full = 0

    episode_entropy = 0
    entropy_history = []

    start = time.time()
    elapsed = 0

    running_reward = 0

    summary = {}
    summary['agent_ID'] = ID
    summary['agent_type'] = 'ERLAM'
    summary['total_episodes'] = episodes_n
    summary['frameskip'] = frameskip

    summary['random_steps'] = agent.AL.epsilon_random_steps
    summary['egreedy_steps'] = agent.AL.epsilon_greedy_steps
    summary['epsilon_max'] = agent.AL.epsilon
    summary['epsilon_min'] = agent.AL.epsilon_min
    summary['epsilon_decay'] = agent.AL.epsilon_decay

    summary['learning_rate'] = agent.AL.learning_rate
    summary['gamma'] = agent.AL.gamma
    summary['lambda'] = agent.AL.lambda_

    summary['batch_size'] = agent.CL.batch_size
    summary['memory_size'] = agent.CL.max_memory
    summary['forgetting'] = agent.CL.forgetting

    summary['dqn_update_freq'] = agent.AL.update_freq
    summary['target_update_freq'] = agent.AL.sync_freq
    summary['graph_update_freq'] = agent.CL.associative_frequency


    obs, rew, done, info = env.step((action[0]*3)+action[1])
    agent_info = info
    visual_obs = obs
    #state = np.array(visual_obs)
    state = visual_obs
    simulation = True

    # Create a new CSV file and write the header row
    with open(savePath+ID+"performance_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episodes", "Steps", "Rewards", "Memories", "Entropy", "Entropy_History"])

    print("STARTING SIMULATION...")
    while simulation:

        # need to reset the environemnt if done
        if done:
            obs = env.reset()

        # TAKE ACTION AT EVERY X STEPS
        if frame_count%frameskip == 0:
            agent_steps += 1
            action = agent.step(state)
            #print("agent entropy:", agent.AL.entropy)
            entropy_history.append(agent.AL.entropy)
            agent.AL.update_epsilon()

        #UPDATE ENVIRONMENT WITH AGENT'S ACTION
        obs, rew, done, info = env.step((action[0]*3)+action[1])
        # NEW STEP
        frame_count += 1

        # GET ENV INFORMATION
        agent_info = info
        next_state = np.expand_dims(obs, axis=0)
        #next_state = np.array(new_obs)
        #next_state = new_obs
        agent_done = done
        reward = rew

        # ADAPT REWARDS
        episode_reward += reward
        rwd = episode_reward if (reward > 0.) else reward

        # STORE EXPERIENCE IN MEMORY
        if (frame_count%frameskip == 0) or (reward > 0.):
            ac_indx = int(action[0]*action_space[0] + action[1])    # convert action[x,x] into an integer for storage
            agent.store_exp(state, ac_indx, rwd, next_state, agent_done)
            state = next_state
            #In reference to the paper, perform lines 11-15 of Algorithm 2
            #agent.update_DQN()
            if (reward > 0.): # PREVIOUSLY
                print ("N ACTIONS TO REACH REWARD:", agent_steps)
                print('FINAL SCORE: ', episode_reward)

        # TRAIN DQN EVERY 4 ACTIONS TAKEN
        if frame_count%(frameskip*agent.AL.update_freq) == 0:
            #In reference to the paper, perform lines 11-15 of Algorithm 2
            agent.update_DQN()
            #pass

        # TRAIN DQN EVERY 4 ACTIONS TAKEN
        if frame_count%(frameskip*agent.AL.sync_freq) == 0:
            agent.AL.update_target()

        # END OF EPISODE
        if agent_done:
            episode_count += 1
            print ("EPISODE "+ str(episode_count) + " DONE!")

            elapsed = time.time() - start
            start = time.time()
            print ("TIME ELAPSED: "+ str(elapsed))

            if (episode_count%1 == 0):
                #In reference to the paper, perform lines 17-21 of Algorithm 2
                agent.update_memory(episode_count)

                #In reference to the paper, perform lines 11-15 of Algorithm 2
                #agent.update_DQN()

            #  LOG EPISODE DATA
            episode_frames = frame_count - episode_frames
            total_reward += episode_reward
            episode_memories = agent.CL.get_memory_length()
            episode_entropy = np.mean(entropy_history)

            # Write the performance data to the CSV file
            with open(savePath+ID+"performance_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                #writer.writerow(["Episodes", "Steps", "Rewards", "Memories", "Entropy"])
                writer.writerow([episode_count, episode_frames, episode_reward, episode_memories, episode_entropy, entropy_history])

            if episode_count >= max_episodes:
                print ('Simulation completed!')
                simulation = False
                summary['total_reward'] = total_reward
                summary['total_frames'] = frame_count

            # RESET AND RESTART
            episode_reward = 0
            agent_steps = 0
            entropy_history = []

            if simulation:
                if len(arenas) > 1: 
                    if capsule: 
                        arena_config_in = '/root/capsule/code/sec/data/utilities/arenas/'+arenas[episode_count%len(arenas)]
                    else:
                        arena_config_in = './utilities/arenas/'+arenas[episode_count%len(arenas)]
                    env._env.reset(arenas_configurations=arena_config_in)
                else:
                    env.reset()
                # need to reset the environemnt if done
                if done:
                    obs = env.reset()
                obs, rew, done, info = env.step(0)

    save_summary(savePath, ID, summary)

    return total_reward

def save_summary(savePath, ID, summary):
    with open(savePath+ID+'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
