import numpy as np
import pickle as pkl 

class MFEC(object):

    def __init__(self, discount=1, k=5, ltm=100, pl=20, forget="NONE", estimation=False, load_ltm=False):
        self.discount = discount
        self.k = k
        self.pl = pl  # pl = prototype length 
        self.nl = ltm # LTM buffer capacity: Total n of sequences stored in LTM
        self.forget = forget # can be "FIFO", "SING" or "PROP"
        self.fgt_ratio = 0.1
        self.estimation = estimation

        self.entropy = 0.
        self.memory_full = False

        print("Discount rate: ", self.discount)
        print("K neighbors: ", self.k)
        print("LTM length: ", self.nl)
        print("Forgetting: ", self.forget)
        print("Estimation: ", self.estimation)

        self.action_space = [3, 3]
        self.action = np.array([-1., -1.])
        self.Q_EC = [[[np.zeros(pl).tolist()], [-10]] for _ in range(self.action_space[0]*self.action_space[1])] 

        if load_ltm: self.load_LTM()

    def action_selection(self, Q_s):
        self.compute_entropy(Q_s)
        ac_indx = np.argmax(Q_s)
        action = [int(ac_indx/self.action_space[0]), int(ac_indx%self.action_space[1])]
        return action

    def compute_entropy(self, policy):
        # Entropy of the prob distr for policy stability. (The sum of the % distribution multiplied by the logarithm -in base 2- of p)
        q = np.ravel(policy)
        # sofmax function corrected for large numbers
        #q = np.exp(q - np.max(q)) / np.exp(q - np.max(q)).sum()
        q = np.exp(q) / np.sum(np.exp(q))
        #print ("POLICY: ", q)
        #print ("PROBS SUM: ", np.sum(q))
        qlog = np.log2(q)
        qlog = np.nan_to_num(qlog)
        qqlog = q*qlog
        qsum = -np.sum(qqlog)
        self.entropy = qsum
        #print ("ENTROPY: ", self.entropy)

    def estimate_return(self, prototype):
        """
        MFEC: updates the action-value function Q_ac and returns it.

        Args:
        state (int): the current state identifier
        Q_ac (ndarray): current action-value function Q_ac of shape (n_states, n_qvalues)

        Returns:
        ndarray: the updated action-value function Q_ac of shape (n_states, n_qvalues)

        """
        # 1. Check if its a new state
        # 2. If it is: Get the k nearest states
        # 3. Calculate the q estimate by averaging the value of the k nearest states
        state = prototype.tolist()
        Q_s = []
        q = -10

        for i, Q_ac in enumerate(self.Q_EC):
            # check if the current state-action couplet is in memory
            if state not in Q_ac[0]:
                # first, calculate which are the closest states in memory (based on Euclidean distance btw state and memories)
                #distances = np.sum(np.abs(state - Q_ac[0]), axis=2) / len(state)
                #Q_ac_reshaped = np.array(Q_ac[0]).reshape(-1, 100)
                #print("state shape", np.array(state).shape)
                #print("Q_ac[0] shape", np.array(Q_ac[0]).shape)
                #print("Q_ac[0][0] shape", Q_ac[0][0].shape)
                distances = np.linalg.norm(state - np.array(Q_ac[0]), axis=1)
                # get an index of the closest neighbors
                #nearest_neighbor_ids = distances.argsort()[:self.k] if len(distances) > self.k else distances.argsort()
                nearest_neighbor_ids = distances.argsort()[:self.k]
                # get q values of closest neighbors
                k_qs = [Q_ac[1][k] for j, k in enumerate(nearest_neighbor_ids)]
                # get the q estimate for the new state
                q = np.mean(k_qs)
                # UNCERTAIN ABOUT THIS: introduce the new state and q estimate on the action-value buffer? or just use estimation?
                if self.estimation:
                    self.check_memory_space(Q_ac,i)
                    Q_ac[0].append(state)
                    Q_ac[1].append(q)
                # return the updated action-value buffer
            else:
                q = Q_ac[1][Q_ac[0].index(state)]

            Q_s.append(q)

        return Q_s

    def value_update(self, prototype, action, reward):
        #print("UPDATING MFEC Q VALUES")
        """
        MFEC: updates the value function Q_EC and returns it.

        Args:
        state (vector): the current state identifier
        action (int): the action taken
        reward (float): the reward received (can be negative)
        Q_EC (ndarray): current value function of shape (n_actions, n_states, n_qvalues)

        Returns:
        ndarray: the updated action-value function Q_EC of shape (n_states, n_qvalues)

        """
        state = prototype.tolist()
        #print("action: ", action)

        for i, Q_ac in enumerate(self.Q_EC):
            if i == action:
                # check if the current state-action couplet is in memory...
                if state not in Q_ac[0]:
                    #print ("ADDING MEMORY!")
                    #print("len Q_ac", len(Q_ac[1]))
                    #print("self.nl", self.nl)
                    # if it's not there, introduce the new state and q estimate on the action-value buffer
                    if (len(Q_ac[1]) < self.nl):
                    # only add if the memory is not full
                        #print ("ADDING NEW MEMORY!")
                        Q_ac[0].append(state)
                        Q_ac[1].append(reward)
                        self.check_memory_space(Q_ac,i)
                else:
                    # if it is, update the q value by picking the max value between the stored q and the received reward
                    Q_ac[1][Q_ac[0].index(state)] = max(Q_ac[1][Q_ac[0].index(state)], reward)
                    # now move the updated state-value to the last position of the buffer
                    Q_ac[1].append(Q_ac[1].pop(Q_ac[0].index(state)))
                    Q_ac[0].append(Q_ac[0].pop(Q_ac[0].index(state)))

    def check_memory_space(self, Q_ac, i):
        # Check if memory is full
        if (len(Q_ac[1]) >= self.nl):
            if self.memory_full == False:
                print ("ACTION BUFFER IS FULL!")
                #print ("NUMBER OF MEMORIES STORED: ", len(Q_ac[1]))
                self.memory_full = True

            #print ("NUMBER OF MEMORIES STORED: ", len(Q_ac[1]))
            if self.forget != "NONE":
                #print("FORGETTING ACTIVATED...")
                #print ("CURRENT LTM rewards: ", self.LTM[2])
                self.forget_memory(Q_ac, i)

    def forget_memory(self, Q_ac, i):
        # Remove sequences when action buffer is full
        #print ("MEMORY before FORGETTING", Q_ac[1])
        if self.forget == "FIFO":
            Q_ac[0] = np.delete(np.array(Q_ac[0]),0,0).tolist()
            Q_ac[1] = np.delete(np.array(Q_ac[1]),0,0).tolist()
            #print ("LEAST USED MEMORY FORGOTTEN")
            #print ("MEMORY after FORGETTING", Q_ac[1])
            #print ("MEMORY after FORGETTING", self.Q_EC[i][1])

    def get_LTM_length(self):
        ltm_len = 0
        for Q_ac in self.Q_EC:
            ltm_len += len(Q_ac[1])
        return ltm_len

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

