#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Main test file

Possible candidate environments(Continual action space and continual state space):
LunarLanderContinuous-v2
BipedalWalker-v2
BipedalWalkerHardcore-v2
CarRacing-v0
"""

import numpy as np
import tensorflow as tf
import gym
import sys
import getopt

from Brain import Brain
from FourRoom import FourRoom

LEARNING_RATE = 1e-3

ALPHA = 0.9
GAUSS_DECAY = 0.9999
BATCH_SIZE = 32

# trace of some variables
SAVE_REWARD_STEP = 200
SAVE_OPTION_STEP = 10000
SAVE_MODEL_STEP = 100
REWARDS_FILE = 'rewards.npy'
DURAS_FILE = 'duras.npy'
OPTIONS_FILE = 'options.npy'


class Comb:

    def __init__(self):

        # test environment
        # self.env = gym.make('Fourrooms-v1')
        self.env = gym.make('LunarLanderContinuous-v2')
        # self.env = gym.make('MountainCarContinuous-v0')
        # self.env = gym.make('Pendulum-v0')

        # seed the env
        self.seed()

        # env_dict
        self.env_dict = {'state_dim': self.env.observation_space.shape[0],
                         'state_scale': self.env.observation_space.high,
                         'action_dim': self.env.action_space.shape[0],
                         'action_scale': self.env.action_space.high}

        # params
        self.params = {'option_num': 4,
                       'delay': 10000,
                       'epsilon': 0.05,
                       'upper_gamma': 0.9,
                       'lower_gamma': 0.9,
                       'upper_capacity': 10000,
                       'lower_capacity': 10000,
                       'upper_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_policy': LEARNING_RATE,
                       'lower_learning_rate_termin': LEARNING_RATE}

        # RL brain
        self.brain = Brain(env_dict=self.env_dict, params=self.params)

        # some variables
        self.gauss = np.ones(self.params['option_num'])
        self.state = np.zeros(self.env_dict['state_dim'])
        self.target = np.zeros(self.env_dict['state_dim'])
        self.start_state = self.state.copy()
        self.option = -1    # current option
        self.option_reward = 0
        self.option_time = 0

        # the array to record reward and options
        self.rewards = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.duras = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.options = np.zeros(SAVE_OPTION_STEP, dtype=np.int32)
        self.reward_counter = 0             # separate reward
        self.option_counter = 0             # separate option
        self.model_counter = 0              # model counter

    def seed(self, seed=42):
        """Wrapper seed the environment for reproducing the result"""

        np.random.seed(seed)
        self.env.seed(np.random.randint(10000))
        tf.set_random_seed(np.random.randint(10000))

    def step(self, action):
        """Wrapper for the environment step, normalized state and action"""

        next_state, reward, done, _ = self.env.step(action * self.env_dict['action_scale'])

        return next_state / self.env_dict['state_scale'], reward, done

    def reset(self):
        """Wrapper for the environment step"""

        self.state, self.target = self.env.reset()
        self.state /= self.env_dict['state_scale']

    def update_params(self, params):
        """Update params"""

        self.params.update(params)
        self.brain = Brain(env_dict=self.env_dict, params=self.params)

    def choose_action(self):
        """Choose action with noise"""

        action = self.brain.l_options[self.option].choose_action(self.state)
        action += np.random.normal(0.0, self.gauss[self.option], self.env_dict['action_dim'])
        action = np.clip(action, -1.0, 1.0)

        return action

    def save_model(self):
        """Save model"""

        self.model_counter += 1
        if self.model_counter == SAVE_MODEL_STEP:
            self.model_counter = 0
            self.brain.save_model()

    def change_option(self):
        """Judge whether change option"""

        self.option_reward = 0
        self.option_time = 0
        self.option = self.brain.u_critic.choose_option(self.state)
        self.start_state = self.state.copy()

    def one_episode(self):
        """Run the process"""

        ep_reward = 0
        t = 0
        self.reset()
        self.change_option()

        while True:

            # choose action with noise
            action = self.choose_action()

            # step the environment
            next_state, reward, done = self.step(action)

            # fill out all buffers
            if self.brain.l_buffers[0].isFilled is False:
                for o in range(self.brain.on):
                    self.brain.l_buffers[o].store(self.state, action, reward, next_state)
            else:
                self.brain.l_buffers[self.option].store(self.state, action, reward, next_state)

            # record the option
            self.record_option()

            # accumulate reward
            ep_reward += reward
            t += 1
            self.option_reward += reward
            self.option_time += 1

            # train lower options and decay the variance
            if self.brain.train_option(BATCH_SIZE, self.option):
                self.gauss[self.option] *= GAUSS_DECAY

            # done
            if done:
                print(" Target:{}. End:{}".format(self.target, next_state * self.env_dict['state_scale']))
                print("timestep={},reward={},gauss={}".format(t, ep_reward, self.gauss[self.option]))
                self.record_reward(ep_reward, t)
                break

            # check whether change option
            self.state = next_state

            # if the termination condition satisfied
            if np.random.uniform() < self.brain.l_options[self.option].get_prob(self.state):
                # calculate option reward
                self.option_reward /= self.option_time ** ALPHA
                self.brain.u_buffer.store(self.start_state, self.option, self.option_reward, self.state)
                self.change_option()

        self.brain.train_policy(BATCH_SIZE)
        self.save_model()

    def many_episodes(self, episode_num):
        """Many episodes"""

        if episode_num == 'inf':
            i = 0
            while True:
                print(i, end='')
                i += 1
                self.one_episode()
        elif isinstance(episode_num, int) and episode_num > 0:
            for i in range(episode_num):
                print(i, end='')
                self.one_episode()
        else:
            print('Invalid input! Please input "inf" or a positive integer!')

    def record_option(self):
        """Record option"""

        # record current option
        self.options[self.option_counter] = self.option
        self.option_counter += 1
        # If the buffer is filled, then save it
        if self.option_counter == SAVE_OPTION_STEP:
            self.option_counter = 0
            with open(OPTIONS_FILE, 'ab') as f:
                np.save(f, self.options)

    def record_reward(self, ep_reward, t):
        """Record reward"""

        # record current option
        self.rewards[self.reward_counter] = ep_reward
        self.duras[self.reward_counter] = t
        self.reward_counter += 1
        # If the buffer is filled, then save it
        if self.reward_counter == SAVE_REWARD_STEP:
            self.reward_counter = 0
            with open(REWARDS_FILE, 'ab') as f:
                np.save(f, self.rewards)
            with open(DURAS_FILE, 'ab') as f:
                np.save(f, self.duras)


def main():
    """Main program"""

    com = Comb()
    com.many_episodes(5000)


def render():
    """render the network"""

    com = Comb()
    com.brain.restore_model(5)

    # render upper critic
    com.brain.u_critic.render()

    # render lower options
    for i in range(com.params['option_num']):
        com.brain.l_options[i].render()


if __name__ == '__main__':

    opts, args = getopt.getopt(sys.argv[1:], "mr")
    for op, value in opts:
        if op == '-m':
            main()
        elif op == '-r':
            render()
