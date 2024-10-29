from numpy import random
import numpy as np
import pickle as pkl
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'
    
def create_arena(seed, number):
    arenas_list = []
    random.seed(seed)
    #print ('Arena Seed: '+str(seed))
    for i in range (number):
        agent_x = random.randint(5, 35)
        agent_z = random.randint(5, 35)
        agent_a = random.randint(0,359)
    
        arena_name = 'random_dac_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    t: 1000
    items:     
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 3, y: 0, z: 3} 
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 250, b: 0}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 37, y: 0, z: 37} 
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 250, b: 250}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 20}
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 250, b: 250}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 37}
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 0, b: 250}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 39}
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 0, b: 150}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 3}
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 0}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 1}
      rotations: [0]
      colors:
      - !RGB {r: 150, g: 0, b: 0}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 3, y: 0, z: 25} 
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 250, b: 0}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 37, y: 0, z: 15} 
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 250}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 7, y: 0, z: 33}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 33, y: 0, z: 7}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: GoodGoalMulti
      sizes:
      - !Vector3 {x: 3, y: 1, z: 1} 
      - !Vector3 {x: 3, y: 1, z: 1} 
      - !Vector3 {x: 1.5, y: 1, z: 1} 
      - !Vector3 {x: 1.5, y: 1, z: 1} 
      - !Vector3 {x: 1.5, y: 1, z: 1} 
      - !Vector3 {x: 1.5, y: 1, z: 1} 
      positions:
      - !Vector3 {x: 3, y: 0, z: 37} 
      - !Vector3 {x: 37, y: 0, z: 3} 
      - !Vector3 {x: 3, y: 0, z: 15} 
      - !Vector3 {x: 37, y: 0, z: 25} 
      - !Vector3 {x: 11.5, y: 0, z: 3} 
      - !Vector3 {x: 28.5, y: 0, z: 37}  
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