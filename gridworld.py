import gym
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

from torch import nn
import torch
import torch.nn.functional as F
from collections import deque
import itertools


from gym import *
from gym.spaces import Discrete, MultiDiscrete, flatdim

from imagemaker import *
rew_buf = []
steps = []
class GridWorld(Env):
    def __init__(self, number_of_robots, number_of_interestpoints, dim, observation_shape):
        super(GridWorld, self).__init__()
        self.observation_shape =observation_shape
        # Hyperparameters
        self.alpha = 0.3
        self.gamma = 0.6

        self.max_epsilon = 0.9
        self.min_epsilon = 0.0025
        self.decay = 0.998
        self.epsilon = 0.9

        self.batch_size = 32
        self.buffer_size = 50000
        self.min_replay_size = 1000
        self.target_update_freq = 1000

        self.dim = dim
        self.number_of_robots = number_of_robots
        self.number_of_interestpoints = number_of_interestpoints
        self.observation_size = 2*self.number_of_robots+3*self.number_of_interestpoints

        self.is_occupied = np.zeros([self.dim, self.dim], dtype=bool)
        self.is_interestpoint = np.zeros([self.dim, self.dim], dtype=bool)
        self.reward = 0
        self.boundarycount = 0

        imagefolder = 'images'
        self.empty_imagefolder(imagefolder)

        self.robots = []
        for i in range(number_of_robots):
            position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])
            self.robots.append(Robot(i, position, self.dim))
            print('Robot ID = {}: Robot Starting Position = [{}, {}]'.format(i, position[0], position[1]))

        self.interestpoints = []
        for i in range(number_of_interestpoints):
            #random.seed(i+10)
            position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])
            self.interestpoints.append(InterestPoint(i, position, self.dim))
            print('Interest Point ID = {}: Starting Position = [{}, {}]'.format(i, position[0], position[1]))

        self.observation_space = self.build_observation_space()

        self.number_of_actions = 4
        self.list_of_actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        state_space_size = ((self.dim*self.dim)**self.number_of_robots)*((2*(self.dim*self.dim))**self.number_of_interestpoints)
        self.action_space_size = self.number_of_actions**self.number_of_robots
        self.state_visits = np.zeros(state_space_size)

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.rew_buffer = deque([0.0], maxlen=100)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('running on the GPU')
        else:
            device = torch.device('cpu')
            print('running on the CPU')

        self.online_net = Network(self.observation_shape, self.action_space_size).to(device)
        self.target_net = Network(self.observation_shape, self.action_space_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

    def build_observation_space(self):
        """
        robot_positions_x = np.asarray([robot.position[0] for robot in self.robots])
        robot_positions_y = np.asarray([robot.position[1] for robot in self.robots])
        IP_positions_x = np.asarray([interestpoint.position[0] for interestpoint in self.interestpoints])
        IP_positions_y = np.asarray([interestpoint.position[1] for interestpoint in self.interestpoints])
        IP_visit = np.asarray([interestpoint.visited for interestpoint in self.interestpoints])

        return np.concatenate((robot_positions_x, robot_positions_y, IP_positions_x, IP_positions_y, IP_visit), axis=None)
        """
        obs = np.zeros((3, self.dim, self.dim), dtype=np.float32)
        return obs

    def reset(self):
        for robot in self.robots:
            robot.position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])


        for interestpoint in self.interestpoints:
            interestpoint.visited = False
            interestpoint.position = np.array([random.randint(0, self.dim - 1), random.randint(0, self.dim - 1)])

    #multi-agent
    def n_to_base(self, n, base):
        n_base = np.array([])
        while n != 0:
            n_base = np.append(n_base, n % base)
            n = n // base

        n_base[:] = n_base[::-1]

        prefix_length = self.number_of_robots - len(n_base)
        prefix = np.zeros(prefix_length)
        actions = np.append(prefix, n_base)
        return actions

    #multi-agent
    def flatten_multidimensional_index(self):
        index = 0
        for robot in self.robots:
            x, y = robot.position
            index += ((self.dim*self.dim)**robot.name)*(x*self.dim+y)

        for interestpoint in self.interestpoints:
            x, y = interestpoint.position
            if interestpoint.visited:
                hash_IP_position_visited = (x*self.dim+y)+(self.dim*self.dim)
            else:
                hash_IP_position_visited = x*self.dim+y

            index += (self.dim*self.dim)**(interestpoint.name+self.number_of_robots)*hash_IP_position_visited

        return index

    def compute_reward(self):
        self.is_occupied = np.zeros([self.dim, self.dim], dtype=bool)
        self.is_interestpoint = np.zeros([self.dim, self.dim], dtype=bool)
        reward = 0

        for interestpoint in self.interestpoints:
            if not interestpoint.visited:
                self.is_interestpoint[tuple(interestpoint.position)] = True

        for robot in self.robots:
            if self.is_occupied[tuple(robot.position)]:
                reward -= 10
            elif self.is_interestpoint[tuple(robot.position)]:
                reward += 20
            elif robot.boundary:
                reward -= 10
                robot.boundary = False
            else:
                reward -= 1

            self.is_occupied[tuple(robot.position)] = True

        self.reward = reward

    def random_action(self):
        action_space_size = self.number_of_actions ** self.number_of_robots
        action_space = np.array(range(action_space_size))
        action_index = np.random.choice(action_space)
        return action_index

    def epsilon_greedy_action(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        return self.online_net.act(obs)

    def terminal_condition(self):
        self.is_occupied = np.zeros([self.dim, self.dim], dtype=bool)

        for robot in self.robots:
            self.is_occupied[tuple(robot.position)] = True

        for interestpoint in self.interestpoints:
            if self.is_occupied[tuple(interestpoint.position)]:
                interestpoint.visited = True

        for interestpoint in self.interestpoints:
            if not interestpoint.visited:
                return False

        return True

    def step(self, actions):
        for robot in self.robots:
            #print('Position Before = {} : Action = {}'.format(robot.position, actions))
            robot.move(actions[robot.name])
            #print('Position After = {}'.format(robot.position))
        self.compute_reward()

    def train(self, number_of_episodes):
        filehandler_state = open('state', 'wb')
        filehandler_convergence = open('convergence', 'wb')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        #Initialize Replay Buffer
        self.reset()
        obs = self.observation_space
        for _ in range(self.min_replay_size):
            action_index = self.random_action()
            actions = self.n_to_base(action_index, self.number_of_actions)
            self.step(actions)
            reward = self.reward
            self.observation_space = self.build_observation_space()
            new_obs = self.observation_space
            done = self.terminal_condition()
            transition = (obs, action_index, reward, done, new_obs)
            #print('Observation = {}: Action = {}: New Observation = {}'.format(obs, actions, new_obs))
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                self.reset()
                obs = self.observation_space

        step = 0

        for episode in range(number_of_episodes):
            self.reset()
            obs = self.observation_space
            reward = 0
            episode_reward = 0

            #new_state_index = 0
            #if episode < int(round(0.5*number_of_episodes, 0)):
            #    self.epsilon = epsilon[episode]
            #else:
            #    self.epsilon = 0.05

            self.epsilon = max(self.min_epsilon, self.epsilon*self.decay)

            #print('Episode = {}: Epsilon = {}'.format(episode, self.epsilon))

            while not self.terminal_condition():
                action_index = self.epsilon_greedy_action(obs)

                #convert action index to multi-dimensional action vector
                actions = self.n_to_base(action_index, self.number_of_actions)

                if episode == number_of_episodes - 1:
                    pickle.dump([episode, step, self.robots, self.interestpoints], filehandler_state)

                self.step(actions)

                #for robot in self.robots:
                #    print('Robot ID: {} Robot Position: {}'.format(robot.name, robot.position))

                reward = self.reward
                self.observation_space = self.build_observation_space()
                new_obs = self.observation_space
                done = self.terminal_condition()

                transition = (obs, action_index, reward, done, new_obs)
                self.replay_buffer.append(transition)
                obs = new_obs

                transitions = random.sample(self.replay_buffer, self.batch_size)

                episode_reward += reward
                rew_buf.append(episode_reward)
                steps.append(step)

                if done:
                    self.rew_buffer.append(episode_reward)
                    episode_reward = 0

                obses = np.asarray([t[0] for t in transitions])
                actions = np.asarray([t[1] for t in transitions])
                rews = np.asarray([t[2] for t in transitions])
                dones = np.asarray([t[3] for t in transitions])
                new_obses = np.asarray([t[4] for t in transitions])

                obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
                rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1).to(device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
                new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)

                target_q_values = self.target_net(new_obses_t)
                #print('Target Q Values = ', target_q_values)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rews_t + self.gamma * (1 - dones_t) * max_target_q_values

                q_values = self.online_net(obses_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                loss = nn.functional.smooth_l1_loss(action_q_values, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1

                if step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if step % 1000 == 0:
                    print()
                    print('Step', step)
                    print('Episode', episode)
                    print('Avg Rew', np.mean(self.rew_buffer))
                    print('Epsilon', self.epsilon)

            pickle.dump([episode, step, self.robots, self.interestpoints], filehandler_state)

        filehandler_convergence.close()
        filehandler_state.close()

        filehandler_statevisit = open('statevisit', 'wb')
        pickle.dump(self.state_visits, filehandler_statevisit)
        filehandler_statevisit.close()

    def evaluate(self, filename_qtable):
        self.reset()
        step = 0
        episode = 0

        filehandler_qtable = open(filename_qtable, 'rb')
        filehandler_evaluation = open('evaluation', 'wb')
        [self.qtable, interestpoints, number_of_robots, dim] = pickle.load(filehandler_qtable)

        while not self.terminal_condition():
            state_index = self.flatten_multidimensional_index()
            action_index = self.greedy_action(state_index)
            actions = self.n_to_base(action_index, self.number_of_actions)
            self.step(actions)
            step += 1

            pickle.dump([episode, step, self.robots, self.interestpoints], filehandler_evaluation)

        filehandler_qtable.close()
        filehandler_evaluation.close()

    def visualize_training(self, filename, number_of_episodes):
        filehandler = open(filename, 'rb')
        while True:
            try:
                [episode, step, robots, interestpoints] = pickle.load(filehandler)
                if episode == number_of_episodes-1:
                    image_maker(step, robots, interestpoints, self.dim)
            except EOFError:
                break

        movie_maker()

    def visualize(self, filename):
        filehandler = open(filename, 'rb');
        while True:
            try:
                [episode, instant, robots, interestpoints] = pickle.load(filehandler)
                image_maker(instant, robots, interestpoints, self.dim)
            except EOFError:
                break

        movie_maker()

    def empty_imagefolder(self, imagefolder):
        for filename in os.listdir(imagefolder):
            file_path = os.path.join(imagefolder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s.  Reason: %s' % (file_path, e))


class Point(object):
    def __init__(self, name, position, dim):
        self.name = name
        self.dim = dim
        self.set_position(position)

    def set_position(self, position):
        temp_x, temp_y = position
        x_min, x_max, y_min, y_max = [0, self.dim-1, 0, self.dim-1]
        x = self.clamp(temp_x, x_min, x_max)
        y = self.clamp(temp_y, y_min, y_max)
        self.position = np.array([x, y])

    def get_position(self):
        return self.position

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Robot(Point):
    def __init__(self, name, position, dim):
        super(Robot, self).__init__(name, position, dim)

        self.action_key = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.boundary = False

    def move(self, action):
        heading = 0
        if action == 0:
            heading = np.array([0, 1])
        elif action == 1:
            heading = np.array([0, -1])
        elif action == 2:
            heading = np.array([-1, 0])
        elif action == 3:
            heading = np.array([1, 0])

        position = self.position
        position += heading
        self.set_position(position)
        if np.linalg.norm(position-self.position) > 0.01:
            self.boundary = True

    def get_action_meanings(self, action):
        return self.action_key[action]


class InterestPoint(Point):
    def __init__(self, name, position, dim):
        super(InterestPoint, self).__init__(name, position, dim)

        self.visited = False

"""
class Network(nn.Module):
    def __init__(self, observation_size, action_space_size):
        super().__init__()

        in_features = int(observation_size)

        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space_size)

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, action_space_size)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

        #return self.net(x)

    def act(self, obs):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action_index = max_q_index.detach().item()

        return action_index
"""
class Network(nn.Module, GridWorld):
    def __init__(self, observation_shape,action_space_size):
        super().__init__()

        #self.conv1 = nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        #self.fc_input_dim = self._calculate_fc_input_dim(observation_shape)

        #self.fc1 = nn.Linear(self.fc_input_dim, 512)
        #self.fc2 = nn.Linear(512, action_space_size)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 32, 64)
        self.fc2 = nn.Linear(64, action_space_size)

    def _calculate_fc_input_dim(self, observation_shape):
        dummy_input = torch.zeros(1, *observation_shape)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    """"
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    """
    """
    def act(self, obs):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
        q_values = self(obs_t)
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action_index = max_q_index.detach().item()

        return action_index
        """
    def act(self, obs):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        # Add a channel dimension
        obs = np.expand_dims(obs, axis=0) # For numpy arrays
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)

        # If your input is already a torch tensor, you can use unsqueeze:
        # obs_t = obs_t.unsqueeze(0).to(device)

        q_values = self(obs_t.unsqueeze(0)) # Add a batch dimension
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action_index = max_q_index.detach().item()

        return action_index
