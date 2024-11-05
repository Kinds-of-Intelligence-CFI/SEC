from numpy import random
import os

if os.name == 'nt': # Windows
    basePath = './utilities/arenas/'
else:
    basePath = '../'

def create_arena(seed, number):
    arenas_list = []
    random.seed(seed)
    start_pos = ['{x: 3, y: 0, z: 37}','{x: 37, y: 0, z: 37}','{x: 3, y: 0, z: 3}','{x: 37, y: 0, z: 3}']
    for i in range (number):
        file_name = 'random_agent_'+str(i)+'.yaml'
        content = '''!ArenaConfig
arenas:
  0: !Arena
    t: 1000
    items:     
    - !Item # blue square top midl
      name: Wall 
      sizes:
      - !Vector3 {x: 2, y: 5, z: 8}
      positions:
      - !Vector3 {x: 19, y: 0, z: 36}
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 0, b: 250}
    - !Item # red square top midr
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 5, z: 8}
      positions:
      - !Vector3 {x: 21, y: 0, z: 36} 
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 0}
    - !Item # red square bot midl
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 5, z: 8}
      positions:
      - !Vector3 {x: 19, y: 0, z: 4}
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 0}
    - !Item # blue square bot midr
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 5, z: 8}
      positions:
      - !Vector3 {x: 21, y: 0, z: 4} 
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 0, b: 250}
    - !Item  # cian wall left mid
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 24}
      positions:
      - !Vector3 {x: 3, y: 0, z: 20} 
      rotations: [0]
      colors:
      - !RGB {r: 0, g: 250, b: 250}
    - !Item # yellow wall right mid
      name: Wall
      sizes:
      - !Vector3 {x: 2, y: 2, z: 24}
      positions:
      - !Vector3 {x: 37, y: 0, z: 20} 
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 250, b: 0}
    - !Item # wall top left
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 33, y: 0, z: 7}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item # wall top right
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 33, y: 0, z: 33}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item # wall midtop center
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 1}
      positions:
      - !Vector3 {x: 20, y: 0, z: 25}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}

    - !Item # wall midtop center
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 1}
      positions:
      - !Vector3 {x: 20, y: 0, z: 26}
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 250}
    - !Item # wall midbot center
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 1}
      positions:
      - !Vector3 {x: 20, y: 0, z: 14}
      rotations: [0]
      colors:
      - !RGB {r: 250, g: 0, b: 250}

    - !Item # wall midbot center
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 1}
      positions:
      - !Vector3 {x: 20, y: 0, z: 15}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item # wall bot left
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 7, y: 0, z: 7}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item # wall bot right
      name: Wall
      sizes:
      - !Vector3 {x: 14, y: 5, z: 2}
      positions:
      - !Vector3 {x: 7, y: 0, z: 33}
      rotations: [0]
      colors:
      - !RGB {r: 153, g: 153, b: 153}
    - !Item # green reward mid
      name: GoodGoal
      sizes:
      - !Vector3 {x: 3, y: 1, z: 1} 
      positions:
      - !Vector3 {x: 20, y: 0, z: 20}
    - !Item
      name: Agent
      positions:
      - !Vector3 '''+str(start_pos[random.randint(0,4)])+'''
      rotations: ['''+str(random.randint(0,359))+''']'''

        with open(basePath+file_name,'w') as f:
            f.write(content)
        #print (content)
        arenas_list.append(file_name)
    #print (arenas_list)
    return arenas_list
    
#create_arena(random.randint(10000),3)
#create_arena(3,3)