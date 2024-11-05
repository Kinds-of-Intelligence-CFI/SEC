from numpy import random
import numpy as np
import pickle as pkl
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'
    
def create_arena_thorndike(seed, number):
    arenas_list = []
    random.seed(seed)
    #print ('Arena Seed: '+str(seed))
    for i in range (number):
        agent_x = random.randint(5, 35)
        agent_z = random.randint(5, 35)
        agent_a = random.randint(0,359)
    
        arena_name = 'random_thorndike_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    t: 250
    items:
    - !Item
      name: Wall
      positions:
      - !Vector3 {x: 20, y: 0, z: 20}
      - !Vector3 {x: 20, y: 3.1, z: 20}
      rotations: [0,0]
      sizes:
      - !Vector3 {x: 10, y: 0.1, z: 10}
      - !Vector3 {x: 10, y: 0.1, z: 10}
      colors:
      - !RGB {r: 153, g: 153, b: 153}
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: WallTransparent
      positions:
      - !Vector3 {x: 15.25, y: .1, z: 16.25}
      - !Vector3 {x: 15.25, y: .1, z: 23.75}
      - !Vector3 {x: 20, y: .1, z: 15.25}
      - !Vector3 {x: 24.75, y: .1, z: 20}
      - !Vector3 {x: 20, y: .1, z: 24.75}
      rotations: [0,0,0,0,0]
      sizes:
      - !Vector3 {x: 0.5, y: 3, z: 2.5}
      - !Vector3 {x: 0.5, y: 3, z: 2.5}
      - !Vector3 {x: 9, y: 3, z: 0.5}
      - !Vector3 {x: .5, y: 3, z: 10}
      - !Vector3 {x: 9, y: 3, z: 0.5}
    - !Item
      name: Cardbox1
      positions:
      - !Vector3 {x: 15.25, y: .1, z: 20}
      rotations: [0]
      sizes:
      - !Vector3 {x: .5, y: 1, z: 4.8}
    - !Item
      name: GoodGoal
      positions:
      - !Vector3 {x: 13, y: 0, z: 23}
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
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
