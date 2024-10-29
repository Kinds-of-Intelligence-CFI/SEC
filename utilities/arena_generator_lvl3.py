from numpy import random
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'

def create_arena(seed, number):
    arenas_list = []
    random.seed(seed)
    for i in range (number):
        file_name = 'random_agent_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    t: 0
    items:
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 36}
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 0, b: 200}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 2}
      positions:
      - !Vector3 {x: 20, y: 0, z: 4}
      rotations: [0]
      colors:
      - !RGB {r: 200, g: 0, b: 0}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 16, y: 5, z: 2}
      positions:
      - !Vector3 {x: 8, y: 0, z: 33}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 16, y: 5, z: 2}
      positions:
      - !Vector3 {x: 32, y: 0, z: 33}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 16, y: 5, z: 2}
      positions:
      - !Vector3 {x: 8, y: 0, z: 7}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item
      name: Wall
      sizes:
      - !Vector3 {x: 16, y: 5, z: 2}
      positions:
      - !Vector3 {x: 32, y: 0, z: 7}
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
      positions:
      - !Vector3 {x: 3, y: 0, z: 37} 
      - !Vector3 {x: 37, y: 0, z: 3} 
      - !Vector3 {x: 3, y: 0, z: 15} 
      - !Vector3 {x: 37, y: 0, z: 25} 
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: 20, y: 0, z: 20}
      rotations: ['''+str(random.randint(0,359))+''']'''

        with open(basePath+file_name,'w') as f:
            f.write(content)
        #print (content)
        arenas_list.append(file_name)
    #print (arenas_list)
    return arenas_list

#create_arena(random.randint(10000),3)
#create_arena(3,3)