#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------
# Author: WangCai
# Date: 08/09/2019
# --------------------

import gym
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import datetime

env = gym.make('Pendulum-v0')


class DDPG:
    def __init__(self):

        # Game Info
        self.algorithm = 'DDPG'
        self.game_name = 'Pendulum'
        self.data_time = str(datetime.date.today())

        self.Num_states = env.observation_space.shape[0]
        self.Num_action = env.action_space.shape[0]
        self.action_max = 2

        # Parameter setting
        self.Gamma = 0.99
        self.Learning_rate_actor = 0.0001
        self.Learning_rate_critic = 0.001

        self.Num_start_training = 100
        self.Num_training = 25000
        self.Num_testing = 10000

        # Experience Replay
        self.Num_batch = 64
        self.Num_replay_memory = 5000

        self.Num_episode_plot = 10

        # Network parameters
        self.first_fc_actor = [self.Num_states, 256]
        self.first_fc_critic = [self.Num_states + self.Num_action, 256]
        self.second_fc = [256, 128]
        self.third_fc_actor = [128, self.Num_action]
        self.third_fc_critic = [128, 1]

        # Input of the Actor network
        self.x = tf.placeholder(tf.float32, shape=[None, self.Num_states])
        # Actor network
        self.Policy = self.Actor(self.x, 'Actor_main')
        self.Policy_target = self.Actor(self.x, 'Actor_target')
        # Input of the Critic network
        self.Critic_inputs = tf.concat([self.Policy, self.x], 1)
        # self.Critic_inputs_target = tf.concat([self.Policy,self.x], 1)
        self.Critic_inputs_target = tf.concat([self.Policy_target, self.x], 1)
        # Critic network
        self.Q_Value = self.Critic(self.Critic_inputs, 'Critic_main')
        self.Q_Value_target = self.Critic(
            self.Critic_inputs_target, 'Critic_target')

        self.Actor_vars = tf.trainable_variables('Actor_main')
        self.Actor_target_vars = tf.trainable_variables('Actor_target')
        self.Critic_vars = tf.trainable_variables('Critic_main')
        self.Critic_target_vars = tf.trainable_variables('Critic_target')
        self.target_critic, self.critic_loss, \
            self.critic_train = self.loss_and_train_critic()
        self.actor_loss, self.actor_train = self.loss_and_train_actor()
        self.sess = self.init_sess()

    def main(self):
        noise = OU_noise(env.action_space, decay_period=self.Num_training)

        state = env.reset()

        noise.reset()

        # Initial parameters
        step = 0
        step_train = 0
        score = 0
        episode = 0

        replay_memory = []

        # Plot
        plt.figure(1)
        plot_x = []
        plot_y = []

        while True:

            if step <= self.Num_start_training:
                progress = 'Exploring'
            elif step <= self.Num_start_training + self.Num_training:
                progress = 'Training'
            elif step < self.Num_start_training + \
                    self.Num_training + \
                    self.Num_testing:
                progress = 'Testing'
            else:
                print('Test is finished!!')
                plt.savefig(
                    './Plot/' +
                    self.data_time +
                    '_' +
                    self.algorithm +
                    '_' +
                    self.game_name +
                    '.png')
                break

            state = state.reshape(-1, self.Num_states)

            # Current Q-network------'Actor_main'
            action = self.sess.run(self.Policy, feed_dict={self.x: state})

            # Add noise
            if progress != 'Testing':
                action = noise.add_noise(action, step_train)

            state_next, reward, terminal, _ = env.step(action)
            state_next = state_next.reshape(-1, self.Num_states)

            # Experience replay
            if len(replay_memory) >= self.Num_replay_memory:
                del replay_memory[0]

            replay_memory.append([state, action, reward, state_next, terminal])

            if progress == 'Training':
                env.render()
                minibatch = random.sample(replay_memory, self.Num_batch)

                # Save the each batch data
                state_batch = [batch[0][0] for batch in minibatch]
                action_batch = [batch[1][0] for batch in minibatch]
                reward_batch = [batch[2][0] for batch in minibatch]
                state_next_batch = [batch[3][0] for batch in minibatch]
                terminal_batch = [batch[4] for batch in minibatch]

                # Update Critic
                y_batch = []
                Q_batch = self.sess.run(
                    self.Q_Value_target, feed_dict={
                        self.x: state_next_batch})

                for i in range(self.Num_batch):
                    if terminal_batch[i]:
                        y_batch.append([reward_batch[i]])
                    else:
                        y_batch.append(
                            [reward_batch[i] + self.Gamma * Q_batch[i][0]])

                _, loss_critic = self.sess.run(
                    [self.critic_train, self.critic_loss], feed_dict={
                        self.target_critic: y_batch,
                        self.x: state_batch,
                        self.Policy: action_batch})

                _, loss_actor = self.sess.run(
                    [self.actor_train, self.actor_loss], feed_dict={
                        self.x: state_batch})

                # Soft Update
                self.Soft_update(self.Actor_target_vars, self.Actor_vars)
                self.Soft_update(self.Critic_target_vars, self.Critic_vars)

                step_train += 1

            # Update parameters at every iteration
            step += 1
            score += reward[0]

            state = state_next

            # Terminal
            if terminal:
                print(
                    'step: ' +
                    str(step) +
                    '/' +
                    'episode: ' +
                    str(episode) +
                    '/' +
                    'state: ' +
                    progress +
                    '/' +
                    'score: ' +
                    str(score))

                # data for plotting
                plot_x.append(episode)
                plot_y.append(score)

                # Plotting
                if episode % self.Num_episode_plot == 0 \
                        and progress == 'Training':
                    plt.xlabel('Episode')
                    plt.ylabel('Score')
                    plt.title('Pendulum_' + self.algorithm)
                    plt.grid(True)
                    plt.plot(
                        np.average(plot_x),
                        np.average(plot_y),
                        hold=True,
                        marker='*',
                        ms=5)

                    plt.draw()
                    plt.pause(0.000001)

                    plot_x = []
                    plot_y = []

                score = 0
                episode += 1

                state = env.reset()
                noise.reset()

    def init_sess(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return sess

    # Initialize weights and bias
    def weight_variable(self, name, shape):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, name, shape):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def Actor(self, x, network_name):
        # Actor Network
        with tf.variable_scope(network_name):
            w_fc1_actor = self.weight_variable('_w_fc1', self.first_fc_actor)
            b_fc1_actor = self.bias_variable(
                '_b_fc1', [self.first_fc_actor[1]])

            w_fc2_actor = self.weight_variable('_w_fc2', self.second_fc)
            b_fc2_actor = self.bias_variable('_b_fc2', [self.second_fc[1]])

            w_fc3_actor = self.weight_variable('_w_fc3', self.third_fc_actor)
            b_fc3_actor = self.bias_variable(
                '_b_fc3', [self.third_fc_actor[1]])

        h_fc1_actor = tf.nn.elu(tf.matmul(x, w_fc1_actor) + b_fc1_actor)
        h_fc2_actor = tf.nn.elu(
            tf.matmul(
                h_fc1_actor,
                w_fc2_actor) +
            b_fc2_actor)

        output_actor = tf.nn.tanh(
            tf.matmul(
                h_fc2_actor,
                w_fc3_actor) +
            b_fc3_actor)
        return self.action_max * output_actor

    def Critic(self, x, network_name):
        with tf.variable_scope(network_name):
            w_fc1_critic = self.weight_variable('_w_fc1', self.first_fc_critic)
            b_fc1_critic = self.bias_variable(
                '_b_fc1', [self.first_fc_critic[1]])

            w_fc2_critic = self.weight_variable('_w_fc2', self.second_fc)
            b_fc2_critic = self.bias_variable('_b_fc2', [self.second_fc[1]])

            w_fc3_critic = self.weight_variable('_w_fc3', self.third_fc_critic)
            b_fc3_critic = self.bias_variable(
                '_b_fc3', [self.third_fc_critic[1]])

        # Critic Network
        h_fc1_critic = tf.nn.elu(tf.matmul(x, w_fc1_critic) + b_fc1_critic)
        h_fc2_critic = tf.nn.elu(
            tf.matmul(
                h_fc1_critic,
                w_fc2_critic) +
            b_fc2_critic)

        output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic
        return output_critic

    def loss_and_train_critic(self):
        target_critic = tf.placeholder(tf.float32, shape=[None, 1])
        critic_loss = tf.losses.mean_squared_error(target_critic, self.Q_Value)
        critic_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.Learning_rate_critic)
        critic_train = critic_optimizer.minimize(
            critic_loss, var_list=self.Critic_vars)

        return target_critic, critic_loss, critic_train

    def loss_and_train_actor(self):
        actor_loss = -tf.reduce_sum(self.Q_Value)
        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.Learning_rate_actor)
        actor_train = policy_optimizer.minimize(actor_loss, var_list=self.Actor_vars)

        return actor_loss, actor_train

    def Soft_update(self, Target_vars, Train_vars, tau=0.001):
        for v in range(len(Target_vars)):
            soft_target = self.sess.run(
                Train_vars[v]) * tau + \
                self.sess.run(Target_vars[v]) * (1 - tau)
            Target_vars[v].load(soft_target, self.sess)

# Ornstein - Uhlenbeck noise
# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


class OU_noise(object):
    def __init__(
            self,
            env_action,
            mu=0.0,
            theta=0.15,
            max_sigma=0.3,
            min_sigma=0.1,
            decay_period=1000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.num_actions = env_action.shape[0]
        self.action_low = env_action.low
        self.action_high = env_action.high
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_actions)

    def state_update(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.num_actions)
        self.state = x + dx

    def add_noise(self, action, training_step):
        self.state_update()
        state = self.state
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * \
            min(1.0, training_step / self.decay_period)
        return np.clip(action + state, self.action_low, self.action_high)


if __name__ == '__main__':
    ddpg = DDPG()
    ddpg.main()
