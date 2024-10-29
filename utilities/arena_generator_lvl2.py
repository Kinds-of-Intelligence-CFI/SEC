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
        file_name = 'random_food_'+str(i)+'.yaml'
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
      - !Vector3 {x: '''+str(random.randint(3, 37))+''', y: 0, z: '''+str(random.randint(3, 37))+'''}
    - !Item
      name: BadGoal
      sizes:
      - !Vector3 {x: 3, y: 1, z: 1} 
      - !Vector3 {x: 3, y: 1, z: 1} 
      - !Vector3 {x: 3, y: 1, z: 1} 
      - !Vector3 {x: 3, y: 1, z: 1} 
      positions:
      - !Vector3 {x: '''+str(random.randint(3, 37))+''', y: 0, z: '''+str(random.randint(3, 37))+'''}
      - !Vector3 {x: '''+str(random.randint(3, 37))+''', y: 0, z: '''+str(random.randint(3, 37))+'''}
      - !Vector3 {x: '''+str(random.randint(3, 37))+''', y: 0, z: '''+str(random.randint(3, 37))+'''}
      - !Vector3 {x: '''+str(random.randint(3, 37))+''', y: 0, z: '''+str(random.randint(3, 37))+'''}
    - !Item
      name: Agent
      positions:
      - !Vector3 {x: '''+str(random.randint(5,35))+''', y: 0, z: '''+str(random.randint(5, 35))+'''}
      rotations: ['''+str(random.randint(0,359))+''']'''

        with open(basePath+file_name,'w') as f:
            f.write(content)
        #print (content)
        arenas_list.append(file_name)
    #print (arenas_list)
    return arenas_list

#create_arena(random.randint(10000),3)
#create_arena(3,3)