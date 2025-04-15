from numpy import random
import numpy as np
import pickle as pkl
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'
    
def create_arena_detour(seed, number):
    arenas_list = []
    random.seed(seed)
    #print ('Arena Seed: '+str(seed))
    for i in range (number):
        agent_x = random.randint(5, 35)
        agent_z = random.randint(5, 35)
        agent_a = random.randint(0,359)
    
        arena_name = 'random_detour_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    timeLimit: 250
    passMark: 0
    items:
    - !Item
      name: Wall
      positions:
      - !Vector3 {x: 20, y: 0, z: 37}
      - !Vector3 {x: 30, y: 0, z: 38.75}
      rotations: [0,0]
      sizes:
      - !Vector3 {x: 35, y: .5, z: 1}
      - !Vector3 {x: 1, y: .5, z: 2.5}
      colors:
      - !RGB {r: 153, g: 153, b: 153}
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: GoodGoal
      positions:
      - !Vector3 {x: 5, y: 0, z: 38}
      sizes:
      - !Vector3 {x: 1, y: 1, z: 1}
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: '''+str(agent_x)+''', y: 0, z: '''+str(agent_z)+'''}
      rotations: ['''+str(agent_a)+''']'''

        with open(basePath+arena_name,'w') as f:
            f.write(content)            
        #print (content)
        arenas_list.append(arena_name)
    #print (arenas_list)
    return arenas_list

#create_arena(random.randint(10000),3)
#create_arena(3,3)
