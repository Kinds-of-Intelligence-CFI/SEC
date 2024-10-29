import numpy as np
import pickle as pkl 

class ContextualLayer(object):

    def __init__(self, pl=20, stm=50, ltm=1000,
        decision_inertia=True, forget="FIFO", load_ltm=False,
        softmax=False, argmax=False, value_function='default',
        alpha_trigger=0.05, coll_threshold_act=0.98, coll_threshold_proportion=0.995, tau=0.9):

        self.ns = stm # STM sequence length
        self.nl = ltm # LTM buffer capacity: Total n of sequences stored in LTM
        self.decision_inertia = decision_inertia   # goal fidelity
        self.forget = forget # can be "FIFO", "SING" or "PROP"
        self.fgt_ratio = 0.1
        print("STM length: ", self.ns)
        print("LTM length: ", self.nl)
        print("Decision Inertia: ", self.decision_inertia)
        print("Forgetting: ", self.forget)
        self.coll_thres_act = coll_threshold_act    # default 0.9
        self.coll_thres_prop = coll_threshold_proportion    #default 0.95
        self.alpha_tr = alpha_trigger       #default 0.005
        self.tau_decay = tau            #deault 0.9
        self.ac = np.array([-1., -1.])
        self.enabled = False
        self.STM = [[np.zeros(pl), np.zeros(2)] for _ in range(self.ns)] # pl = prototype length 
        self.LTM = [[],[],[]] 
        self.tr = []
        self.last_actions_indx = []
        self.selected_actions_indx = []

        self.action_space = [3, 3]
        self.entropy = 0.
        self.value_function = value_function
        self.softmax = softmax
        self.argmax = False
        self.memory_full = False
        # value_functions = ['default', 'noGi', 'noDist', 'noRR', 'soloGi', 'soloDist', 'soloRR']
        print('Value function mode: ', self.value_function)
        print('Softmax mode: ', self.softmax)
        #print('Argmax mode: ', self.argmax)

        if load_ltm: self.load_LTM()

    def advance(self, e, argmax=False):
        self.argmax = argmax

        if len(self.LTM[0]) > 0:

            bias = 1
            if self.decision_inertia:
                bias = np.array(self.tr)
                #print("bias: ", bias)
                #print("bias length: ", len(bias[0])) # proportional to sequence's length, n = LTM sequences

            collectors = (1 - (np.sum(np.abs(e - self.LTM[0]), axis=2)) / len(e)) * bias
            #print ("collectors ", collectors) # proportional to sequence's length, n = LTM sequences
            #print ("collectors length", len(collectors[0])) 
            #print ("collectors relative", collectors/collectors.max())
            #print ("collectors ", collectors.max())

            # Collector values must be above both thresholds (absolute and relative) to contribute to action.
            self.selected_actions_indx = (collectors > self.coll_thres_act) & ((collectors/collectors.max()) > self.coll_thres_prop) # proportional to sequence's length, n = LTM sequences
            #print ("selected_actions_indx ", self.selected_actions_indx)
            #print ("selected_actions_indx length", len(self.selected_actions_indx))
            #print ("selected_actions_indx length [0]", len(self.selected_actions_indx[0]))
            #print ("selected_actions_indx ", self.selected_actions_indx.sum())

            if np.any(self.selected_actions_indx):

                actions = np.array(self.LTM[1])[self.selected_actions_indx]
                # chooose (normalized, or relative) rewards of sequences with actions selected 
                rewards = np.array(self.LTM[2])[(np.nonzero(self.selected_actions_indx)[0])]
                rewards = rewards/rewards.max()
                # choose (normalized) distances of each action selected within its sequence
                distances = (self.ns - np.nonzero(self.selected_actions_indx)[1])/self.ns
                # choose collector info about the actions selected (that take euclidean distance of current state and collector's selected prototypes)
                collectors = collectors[self.selected_actions_indx]

                # map each selected action-vector into a matrix of N dimensions where N are the dimensions of the action space
                m = np.zeros((len(actions), self.action_space[0], self.action_space[1]))
                #m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = ((collectors*rewards)/distances)
                if self.value_function == 'default':
                    #print('COMPUTING ACTIONS CLASSIC SEC...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*(rewards*np.exp(-distances/self.tau_decay))
                if self.value_function == 'noGi':
                    #print('COMPUTING ACTIONS WITHOUT SIMILARITY...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = rewards*np.exp(-distances/self.tau_decay)
                if self.value_function == 'noDist':
                    #print('COMPUTING ACTIONS WITHOUT DISTANCE...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*rewards
                if self.value_function == 'noRR':
                    #print('COMPUTING ACTIONS WITHOUT REWARD...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors*np.exp(-distances/self.tau_decay)
                if self.value_function == 'soloGi':
                    #print('COMPUTING ACTIONS WITH ONLY SIMILARTY...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = collectors
                if self.value_function == 'soloDist':
                    #print('COMPUTING ACTIONS WITH ONLY DISTANCE...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = np.exp(-distances/self.tau_decay)
                if self.value_function == 'soloRR':
                    #print('COMPUTING ACTIONS WITH ONLY RELATIVE REWARD...')
                    m[np.arange(len(actions)), actions[:,0].astype(int), actions[:,1].astype(int)] = rewards

                q = np.sum(m, axis=0) # aggregate action values from retrieved memories
                q = np.ravel(q)

                if self.softmax:
                    # ACCURATE: normalization process of the value table through softmax
                    q = np.exp(q - np.max(q)) / np.exp(q - np.max(q)).sum()
                else:
                    # UNACCURATE (DEFAULT): proportion of being selected based on the action's relative reward based on the stored experiences
                    q = q/q.sum()

                self.compute_entropy(q)

                if self.argmax:
                    # RL STANDARD: ARGMAX
                    ac_indx = np.argmax(q)
                else:
                    # SEC DEFAULT: SAMPLE FROM WEIGHTED PROBABILITY
                    ac_indx = np.random.choice(np.arange(int(self.action_space[0]*self.action_space[1])), p=q)

                self.ac = [int(ac_indx/self.action_space[0]), int(ac_indx%self.action_space[1])]

            else:
                self.ac = [-1, -1]

            self.selected_actions_indx = self.selected_actions_indx.tolist()
            #print ("selected_actions_indx ", self.selected_actions_indx)
            
        return self.ac #* self.enabled

    def compute_entropy(self, policy):
        # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
        #q = np.exp(policy) / np.sum(np.exp(policy))
        # sofmax function corrected for large numbers
        #q = np.exp(q - np.max(q)) / np.exp(q - np.max(q)).sum()
        q = policy
        #print ("POLICY: ", q)
        #print ("PROBS SUM: ", np.sum(q))
        qlog = np.log2(q)
        qlog = np.nan_to_num(qlog)
        qqlog = q*qlog
        qsum = -np.sum(qqlog)
        self.entropy = qsum
        #print ("ENTROPY: ", self.entropy)

    # Couplet expects a list with [prototype, action]; Goal is -1 or 1 indicating aversive or appetitive goal has been reached.
    def update(self, couplet=[], reward=0):

        # Update STM buffer with the new couplet.
        self.STM.append(couplet)
        self.STM = self.STM[1:] # renew the STM buffer by removing the first value of the STM
        #print ("STM: ", self.STM[-1])

        # NEW: Update the last actions index first!
        self.last_actions_indx = np.copy(self.selected_actions_indx).tolist()  # Updates the last action indexes with the current actions indexes.
        #print ("last_actions_indx ", self.last_actions_indx)

        # Update trigger values.
        if (len(self.tr) > 0) and self.decision_inertia:
            self.tr = (np.array(self.tr) * (1. - self.alpha_tr)) + self.alpha_tr  # trigger values decay by default
            self.tr[(self.tr < 1.)] = 1.       # all trigger values below 1 are reset to 1.
            tr_last_actions_indx = np.array(self.last_actions_indx)
            self.tr[tr_last_actions_indx] = 1.    # NEW: the trigger value of previously selected segments are reset to 1!!!
            last_actions_shifted = np.roll(self.last_actions_indx, 1, axis=1) # shift the matrix one step to the right
            last_actions_shifted[:, 0] = False  # set the first value of each sequence to False 

            # NEW: increase ONLY the trigger value of the next element in sequence (after the ones selected before)!
            tr_change_indx = np.array(last_actions_shifted)
            self.tr[tr_change_indx] += 0.01    # NEW: increase by an arbitrary amount (this amount should be tuned or modified).
            self.tr = self.tr.tolist()

            ## TO-DO ADD FORGETTING OF SEQUENCES BASED ON TRIGGER VALUES.

    def updateLTM(self, reward=0):
        # Update LTM if reached goal state and still have free space in LTM.
        if (reward > 0.) and (len(self.LTM[2]) < self.nl):
            print ("GOAL STATE REACHED! REWARD: ", reward)
            self.LTM[0].append([e[0] for e in self.STM])  #append prototypes of STM couplets.
            self.LTM[1].append([a[1] for a in self.STM])  #append actions of STM couplets.
            self.LTM[2].append(reward)
            self.tr.append(np.ones(self.ns).tolist())
            self.selected_actions_indx.append(np.zeros(self.ns, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.ns, dtype='bool').tolist())
            print("Sequences in LTM", len(self.LTM[2]), ", Sequence length:", len(self.STM))

        # Remove sequences when LTM is full
        if (len(self.LTM[2]) >= self.nl) and self.forget != "NONE":
            if self.memory_full == False:
                #print ("LTM IS FULL!")
                self.memory_full = True

            #print ("FORGETTING ACTIVATED...")
            #print ("CURRENT LTM rewards: ", self.LTM[2])

            if self.forget == "FIFO":
                self.LTM[0] = np.delete(np.array(self.LTM[0]),0,0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),0,0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),0,0).tolist()
                self.tr = np.delete(np.array(self.tr),0,0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),0,0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),0,0).tolist()
                #print ("FIRST MEMORY SEQUENCE FORGOTTEN")
                #print ("UPDATED LTM rewards: ", self.LTM[2])
            elif self.forget == "SING":
                idx = np.argsort(self.LTM[2])
                self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0],0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0],0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0],0).tolist()
                self.tr = np.delete(np.array(self.tr),idx[0],0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0],0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0],0).tolist()
                #print ("LOWEST REWARD SEQUENCE FORGOTTEN")
                #print ("UPDATED LTM rewards: ", self.LTM[2])
            elif self.forget == "PROP":
                maxfgt = int(len(self.LTM[2]) * self.fgt_ratio)
                idx = np.argsort(self.LTM[2])
                self.LTM[0] = np.delete(np.array(self.LTM[0]),idx[0:maxfgt],0).tolist()
                self.LTM[1] = np.delete(np.array(self.LTM[1]),idx[0:maxfgt],0).tolist()
                self.LTM[2] = np.delete(np.array(self.LTM[2]),idx[0:maxfgt],0).tolist()
                self.tr = np.delete(np.array(self.tr),idx[0:maxfgt],0).tolist()
                self.selected_actions_indx = np.delete(np.array(self.selected_actions_indx),idx[0:maxfgt],0).tolist()
                self.last_actions_indx = np.delete(np.array(self.last_actions_indx),idx[0:maxfgt],0).tolist()
                #print ("NUMBER OF FORGOTTEN SEQUENCES: ", maxfgt)
                #print ("UPDATED LTM rewards: ", self.LTM[2])

    def refresh(self):
        if (len(self.tr) > 0): 
            self.tr = np.array(self.tr)
            self.tr[:] = 1.0
            self.tr = self.tr.tolist()

    def normalize_vector(self, v):
        v_norm = v / np.max(v)
        v_norm[np.isnan(v_norm)] = 0.
        return v_norm

    def save_LTM(self, savePath, ID, n=1):
        with open(savePath+ID+'ltm'+str(len(self.LTM[2]))+'_'+str(n)+'.pkl','wb') as f:
            pkl.dump(self.LTM, f)

    def load_LTM_memory(self, filename):
        ID = '/LTMs/'+filename
        #ID = '/LTMs/LTM100_N961.pkl'
        # open a file, where you stored the pickled data
        file = open(ID, 'rb')
        # load information from that file
        self.LTM = pkl.load(file)
        print("LTM loaded!! Memories retrieved: ", len(self.LTM[2]))
        for s in (self.LTM[2]):
            self.tr.append(np.ones(self.ns).tolist())
            self.selected_actions_indx.append(np.zeros(self.ns, dtype='bool').tolist())
            self.last_actions_indx.append(np.zeros(self.ns, dtype='bool').tolist())

        file.close()

    def get_LTM_length(self):
        # In number of sequences
        ltm_len =  len(self.LTM[2])
        return ltm_len

    def get_memory_length(self):
        # In single memory units
        memories =  len(self.LTM[2]) * self.ns
        return memories
