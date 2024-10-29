import numpy as np
from numpy import random
from keras.models import load_model


class ReactiveLayer(object):

    def __init__(self):
        # Variables related to random exploration
        self.random_straight_range = 20
        self.random_turn_range = 20
        self.random_straight = random.randint(1, self.random_straight_range)
        self.random_turn = random.randint(1, self.random_turn_range)
        self.random_action = 0
        self.random_movement_counter = 0

    def feed_img(self, obs):
        action = self.network.predict([obs])[0]
        action_vector = np.array([np.argmax(action[:3]), np.argmax(action[3:])], dtype=int)
        return action_vector

    def random_step(self):
        """Generate a series of straight moves followed by a series of turning moves in a random direction"""
        self.random_movement_counter += 1

        # check if series of random actions has been completed and if so reset variables
        if self.random_movement_counter > self.random_turn + self.random_straight:
            # reset variables
            self.random_movement_counter = 0
            self.random_action = random.randint(0, 2)
            self.random_straight = random.randint(1, self.random_straight_range+1)
            self.random_turn = random.randint(1, self.random_turn_range+1)

        # take a series of straight movements
        if self.random_movement_counter <= self.random_straight:
            return [1, 0]

        # take a series of rotation movements
        elif self.random_straight < self.random_movement_counter <= self.random_turn + self.random_straight:
            if self.random_action == 0:
                return [0, 1]
            else:
                return [0, 2]