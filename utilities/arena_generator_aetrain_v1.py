from numpy import random
import numpy as np
import pickle as pkl
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'
    
def create_arena(seed, number,):
    save_data=False
    arenas_list = []
    coordinates = []
    random.seed(seed)
    #print ('Arena Seed: '+str(seed))
    for i in range (number):
        green_x = random.randint(3, 37)
        green_z = random.randint(3, 37)
        agent_x = random.randint(5, 35)
        agent_z = random.randint(5, 35)
        agent_a = random.randint(0,359)
        if save_data: 
          file_name = 'coordinates.pkl'
          data = (green_x, green_z, agent_x, agent_z, agent_a)
          coordinates.append(data)       
        arena_name = 'random_food_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    t: 85
    items:
    - !Item
      name: GoodGoal
      sizes:
      - !Vector3 {x: 3, y: 1, z: 1} 
      positions:
      - !Vector3 {x: '''+str(green_x)+''', y: 0, z: '''+str(green_z)+'''}
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: '''+str(agent_x)+''', y: 0, z: '''+str(agent_z)+'''}
      rotations: ['''+str(agent_a)+''']'''

        with open(basePath+arena_name,'w') as f:
            f.write(content)   
        #print (content)
        arenas_list.append(arena_name)
    if save_data:
      with open(basePath+file_name,'wb') as f:
        pkl.dump(coordinates, f)
        print ('DATA SAVED SUCCESFULLY!')
        #print (coordinates)
    #print (arenas_list)
    return arenas_list

#create_arena(random.randint(10000),3)
#create_arena(3,3)