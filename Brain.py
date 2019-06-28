#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The upper policy and the whole training process defined here
"""

import sys
import numpy as np
import tensorflow as tf

from Option import Option
from LCritic import LCritic
from Buffer import Buffer
from UCritic import UCritic

# the number of models that will be saved


class Brain:

    def __init__(self, env_dict, params):
        """
        option_num, state_dim, action_dim, action_bound, gamma, learning_rate, replacement,
                 buffer_capacity, epsilon
        gamma: (u_gamma, l_gamma)
        learning_rate: (lr_u_policy, lr_u_critic, lr_option, lr_termin, lr_l_critic)
        """

        # session
        self.sess = tf.Session()

        # environment parameters
        self.sd = env_dict['state_dim']
        self.ad = env_dict['action_dim']
        a_bound = env_dict['action_scale']
        assert a_bound.shape == (self.ad,), 'Action bound does not match action dimension!'

        # hyper parameters
        self.on = params['option_num']
        epsilon = params['epsilon']
        u_gamma = params['upper_gamma']
        l_gamma = params['lower_gamma']
        u_capac = params['upper_capacity']
        l_capac = params['lower_capacity']
        u_lrcri = params['upper_learning_rate_critic']
        l_lrcri = params['lower_learning_rate_critic']
        l_lrpol = params['lower_learning_rate_policy']
        l_lrter = params['lower_learning_rate_termin']

        # the frequency of training termination function
        if params['delay'] == 'inf':
            self.delay = -1
        else:
            self.delay = params['delay']

        # Upper critic and buffer
        self.u_critic = UCritic(session=self.sess, state_dim=self.sd, option_num=self.on,
                                gamma=u_gamma, epsilon=epsilon, learning_rate=u_lrcri)
        self.u_buffer = Buffer(state_dim=self.sd, action_dim=1, capacity=u_capac)

        # Lower critic, options and buffer HER
        self.l_critic = LCritic(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                gamma=l_gamma, learning_rate=l_lrcri)
        self.l_options = [Option(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                 ordinal=i, learning_rate=[l_lrpol, l_lrter])
                          for i in range(self.on)]
        self.l_buffers = [Buffer(state_dim=self.sd, action_dim=self.ad, capacity=l_capac)
                          for i in range(self.on)]

        # Initialize all coefficients and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

        self.tc = 0         # counter for training termination
        self.mc = 0         # counter for model

    def train_policy(self, batch_size):
        """Train upper critic(policy)"""

        if self.u_buffer.isFilled:

            # sample batches
            state_batch, option_batch, reward_batch, next_state_batch = self.u_buffer.sample(batch_size)
            # training
            self.u_critic.train(state_batch, option_batch, reward_batch, next_state_batch)

    def train_option(self, batch_size, option):
        """Train option"""

        # only train after the buffer is filled up
        if self.l_buffers[option].isFilled:

            # sample batches
            state_batch, action_batch, reward_batch, next_state_batch = \
                self.l_buffers[option].sample(batch_size)
            next_action_batch = self.l_options[option].get_target_actions(next_state_batch)

            # train lower critic
            self.l_critic.train(state_batch, action_batch, reward_batch, next_state_batch,
                                next_action_batch)

            # get affiliated batch
            q_gradients_batch = self.l_critic.q_gradients(state_batch, action_batch)
            self.tc += 1
            if self.tc == self.delay:
                advantage_batch = self.l_critic.q_batch(state_batch, action_batch) - \
                                  self._value_batch(state_batch)
                self.tc = 0
            else:
                advantage_batch = None


            # train lower options
            self.l_options[option].train(state_batch, q_gradients_batch, advantage_batch)

            return True

        return False

    def pretrain(self):
        """Pretrain ucritic and termination function"""

        self.u_critic.pretrain()
        for i in range(self.on):
            self.l_options[i].pretrain()
            self.l_options[i].render()
        self.save_model()

    def render(self):
        """Render ucritic and termination function"""

        self.u_critic.render()
        for i in range(self.on):
            self.l_options[i].render()

    def save_model(self):
        """Save current model"""

        if self.mc == 0:
            self.saver.save(self.sess, './model/model.ckpt', global_step=0, write_meta_graph=True)
        else:
            self.saver.save(self.sess, './model/model.ckpt', global_step=self.mc, write_meta_graph=False)
        self.mc += 1

    def restore_model(self, order=None):
        """Restore trained model"""

        ckpt = tf.train.get_checkpoint_state('./model/')
        saver = tf.train.import_meta_graph(ckpt.all_model_checkpoint_paths[0] + '.meta')
        if order is None:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(self.sess, ckpt.all_model_checkpoint_paths[order])
            self.mc = order + 1

    def _value_batch(self, state_batch):
        """The upper policy average of Q value for each option
        :return: the value
        """

        batch_size = state_batch.shape[0]
        value_batch = np.zeros((batch_size, 1))
        action_batch = [self.l_options[i].get_actions(state_batch) for i in range(self.on)]
        q_batch = [self.l_critic.q_batch(state_batch, action_batch[i]) for i in range(self.on)]
        distribution_batch = self.u_critic.get_distribution(state_batch)

        # calculate the value function
        for i in range(batch_size):
            for j in range(self.on):
                value_batch[i] += q_batch[j][i] * distribution_batch[i, j]

        return value_batch
