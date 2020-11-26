"""
Adopted from https://github.com/alversafa/option-critic-arch/blob/master/fourrooms.py.

Modified to return one hot encoded states and gym compatible.
"""

import numpy as np
from gym.utils import seeding
from gym import spaces
import gym

class FourRooms(gym.Env):

    def __init__(self, goal=62, p=0, config=1, layout='3roomsh'):
        """
        config -> configouration of the state space
            0 - returns tabular index of the state
            1 - returns one hot encoded vector of the state
            2 - returns matrix form of the state
        """
        if(layout == '4rooms'):
            layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        elif(layout == '3rooms'):
            layout = """\
wwwwwwwwwwwww
w   w   w   w
w   w       w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w   w   w   w
w       w   w
w   w   w   w
w   w   w   w
wwwwwwwwwwwww
"""
        elif(layout == '3roomsh'):
            layout = """\
wwwwwwwwwwwww
w           w
w           w
wwwwwwwww www
w           w
w           w
w           w
w           w
ww wwwwwwwwww
w           w
w           w
w           w
wwwwwwwwwwwww
"""
        elif(layout == 'maze'):
            layout = """\
wwwwwwwwwwwww
w           w
w ww wwwwww w
w w       w w
w w wwwww w w
w w w   w w w
w w   w   www
w w w   w w w
w w wwwww w w
w w       w w
w ww wwwwww w
w           w
wwwwwwwwwwwww
"""
        elif(layout == 'open'):
            layout = """\
wwwwwwwwwwwww
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
w           w
wwwwwwwwwwwww
"""
        else:
            raise
        self.p = p # Stocasticity the environment
        self.config = config
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        self.a_space = np.array([0, 1, 2, 3])
        self.obs_space = np.zeros(np.sum(self.occupancy == 0))

        # Setting the observation space based on the config
        if(config <= 1):
            self.observation_space = spaces.Box(low=np.zeros(np.sum(self.occupancy == 0)), high=np.ones(np.sum(self.occupancy == 0)), dtype=np.uint8)
        elif(config == 2):
            self.observation_space = spaces.Box(low=np.zeros(169), high=np.ones(169), dtype=np.uint8)

        self.action_space = spaces.Discrete(4)
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        # Random number generator
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i,j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k, v in self.tostate.items()}


        self.goal = goal # East doorway
        self.init_states = list(range(self.obs_space.shape[0]))
        self.init_states.remove(self.goal)
        self.updates = 0
        self.horizon = 200


    def render(self, show_goal=True, show_agent=True):
        current_grid = np.array(self.occupancy)
        if(show_agent):
            current_grid[self.current_cell[0], self.current_cell[1]] = -1
        if show_goal:
            goal_cell = self.tocell[self.goal]
            current_grid[goal_cell[0], goal_cell[1]] = -2
        return current_grid

    def render_state(self):
        occupancy = self.occupancy * 0.01
        current_grid = np.array(occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = 1
        goal_cell = self.tocell[self.goal]
        current_grid[goal_cell[0], goal_cell[1]] = -0.01
        return current_grid

    def seed(self, seed=None):
        """
        Setting the seed of the agent for replication
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init=None):
        self.updates = 0
        if(init is None):
            state = self.rng.choice(self.init_states)
        else:
            state = init
        # state = 0 # fix starting state
        self.current_cell = self.tocell[state]
        if(self.config == 0):
            return state
        elif(self.config == 1):
            temp = np.zeros(len(self.obs_space))
            temp[state] = 1
            return temp
        elif(self.config == 2):
            return self.render_state().flatten()
        else:
            raise
            

    def check_available_cells(self, cell):
        available_cells = []

        for action in range(len(self.a_space)):
            next_cell = tuple(cell + self.directions[action])

            if not self.occupancy[next_cell]:
                available_cells.append(next_cell)

        return available_cells
        

    def step(self, action):
        '''
        Takes a step in the environment with 1-self.p probability. And takes a step in the
        other directions with probability self.p with all of them being equally likely.
        '''
        self.updates += 1

        next_cell = tuple(self.current_cell + self.directions[action])

        if not self.occupancy[next_cell]:

            if self.rng.uniform() < self.p:
                available_cells = self.check_available_cells(self.current_cell)
                self.current_cell = available_cells[self.rng.randint(len(available_cells))]

            else:
                self.current_cell = next_cell

        state = self.tostate[self.current_cell]

        # When goal is reached, it is done
        done = state == self.goal

        if(done):
            reward = 0
        else:
            reward = -1

        if(self.updates>=self.horizon):
            reward = -1
            done = True

        if(self.config == 0):
            return state, reward, done, {}
        elif(self.config == 1):
            temp = np.zeros(len(self.obs_space))
            temp[state] = 1
            return temp, reward, done, {}
        elif(self.config == 2):
            return self.render_state().flatten(), reward, done, {}

class FourRoomsMatrix(FourRooms):
    def __init__(self, goal=62, p=0, layout='3roomsh'):
        FourRooms.__init__(self, goal=goal, p=p, config=2, layout=layout)


class FourRoomsNoTerm(FourRooms):
    """
    Environment with no terminal state but with a probability of dying.

    """
    def __init__(self, p=0, dying=0, config=1, layout='3roomsh'):
        FourRooms.__init__(self, p=p, config=config, layout='3roomsh')
        self.dying = dying

    def render(self):
        return FourRooms.render(self, show_goal=False)

    def render_state(self):
        occupancy = self.occupancy * 0.01
        current_grid = np.array(occupancy)
        current_grid[self.current_cell[0], self.current_cell[1]] = 1
        goal_cell = self.tocell[self.goal]
        return current_grid

    def step(self, action):
        '''
        Takes a step in the environment with 1-self.p probability. And takes a step in the
        other directions with probability self.p with all of them being equally likely.
        '''
        self.updates += 1
        reward = 0 # reward is always 0

        next_cell = tuple(self.current_cell + self.directions[action])

        if not self.occupancy[next_cell]:

            if self.rng.uniform() < self.p:
                available_cells = self.check_available_cells(self.current_cell)
                self.current_cell = available_cells[self.rng.randint(len(available_cells))]
            else:
                self.current_cell = next_cell

        state = self.tostate[self.current_cell]

        if(self.rng.uniform() < self.dying): # randomly check if the agent dies
            done = 1
        else:
            done = 0

        if(self.config == 0):
            return state, reward, done, {}
        elif(self.config == 1):
            temp = np.zeros(len(self.obs_space))
            temp[state] = 1
            return temp, reward, done, {}
        elif(self.config == 2):
            return self.render_state().flatten(), reward, done, {}


class FourRoomsMatrixNoTerm(FourRoomsNoTerm):
    def __init__(self, p=0, dying=0.01, layout='3roomsh'):
        FourRoomsNoTerm.__init__(self, p=p, dying=dying, config=2, layout='3roomsh')