#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------
# Author: SiyuZhou
# Date: 08/09/2019
# --------------------

# Import modules
import tensorflow as tf
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import gym
import os


class ActorCritic:
    def __init__(self):

        # Game Info
        self.algorithm = 'ActorCritic'
        self.game_name = 'CartPole'

        self.Num_features = 4
        self.Num_action = 2

        self.Num_plot_episode = 1

        self.Gamma = 0.98

        # Date of training
        self.date_time = str(datetime.date.today())

        self.Is_solved = False

        self.first_fc = [self.Num_features, 256]
        self.second_fc = [256, 128]
        self.third_fc_actor = [128, self.Num_action]
        self.third_fc_critic = [128, 1]

        self.Learning_rate_actor = 0.0002
        self.Learning_rate_critic = 0.001

        # Initialization
        self.output_actor, self.output_critic = self.network()
        self.target_critic, self.train_critic = self.loss_and_train_critic()
        self.action_actor, self.advantage_actor, \
            self.train_actor = self.loss_and_train_actor()
        self.sess, self.saver = self.init_sess()

    def main(self):
        env = gym.make('CartPole-v0')

        # Plot
        plt.figure(1)
        plot_x = []
        plot_y = []

        score = 0
        running_reward = []
        episode = 1

        state = env.reset()

        action = env.action_space.sample()
        state, reward, terminal, info = env.step(action)

        while True:
            env.render()

            state_feed = np.reshape(state, (1, 4))

            # ----------Actor network---->action---------#
            Policy = self.output_actor.eval(
                feed_dict={self.x: state_feed}).flatten()
            action_step = np.random.choice(self.Num_action, 1, p=Policy)[0]
            action = np.zeros([1, self.Num_action])
            action[0, action_step] = 1
            # -------------------------------------------#

            state_next, reward, terminal, info = env.step(action_step)
            state_next_feed = np.reshape(state_next, (1, 4))

            # ----------Critic network---->v和v_next-----#
            value = self.output_critic.eval(feed_dict={self.x: state_feed})[0]
            value_next = self.output_critic.eval(
                feed_dict={self.x: state_next_feed})[0]
            # -------------------------------------------#

            if terminal:
                advantage = reward - value
                target = [reward]
            else:
                advantage = (reward + self.Gamma * value_next) - value
                target = reward + self.Gamma * value_next

            self.train_actor.run(
                feed_dict={
                    self.action_actor: action,
                    self.advantage_actor: advantage,
                    self.x: state_feed})
            self.train_critic.run(
                feed_dict={
                    self.target_critic: target,
                    self.x: state_feed})

            # Update
            score += reward
            state = state_next

            # Terminal
            if terminal:
                running_reward.append(score)
                plot_x.append(episode)
                plot_y.append(score)

                # Plot average score
                if episode % self.Num_plot_episode == 0:
                    avg_score = np.mean(running_reward[-25:])
                    print(
                        "running_reward[-25:]={}".format(running_reward[-25:]))
                    print(
                        "Episode: " +
                        str(episode) +
                        " Score: " +
                        str(score) +
                        " Avg_score:" +
                        str(avg_score))
                    if avg_score >= 200:
                        print("Solved!")
                        self.Is_solved = True
                        plt.savefig(
                            './Plot/' +
                            self.date_time +
                            '_' +
                            self.algorithm +
                            '_' +
                            self.game_name +
                            '.png')
                        break
                    plt.xlabel('Episode')
                    plt.ylabel('Score')
                    plt.title('Cartpole_' + self.algorithm)
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

            if self.Is_solved:
                break

    def init_sess(self):
        config = tf.ConfigProto()
        sess = tf.InteractiveSession(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        return sess, saver

    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    def network(self):
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, shape=[None, self.Num_features])

        with tf.variable_scope('Actor'):
            w_fc1_actor = self.weight_variable(self.first_fc)
            b_fc1_actor = self.bias_variable([self.first_fc[1]])

            w_fc2_actor = self.weight_variable(self.second_fc)
            b_fc2_actor = self.bias_variable([self.second_fc[1]])

            w_fc3_actor = self.weight_variable(self.third_fc_actor)
            b_fc3_actor = self.bias_variable([self.third_fc_actor[1]])

        h_fc1_actor = tf.nn.relu(tf.matmul(self.x, w_fc1_actor) + b_fc1_actor)
        h_fc2_actor = tf.nn.relu(
            tf.matmul(
                h_fc1_actor,
                w_fc2_actor) +
            b_fc2_actor)
        output_actor = tf.nn.softmax(
            tf.matmul(
                h_fc2_actor,
                w_fc3_actor) +
            b_fc3_actor)

        with tf.variable_scope('Critic'):
            w_fc1_critic = self.weight_variable(self.first_fc)
            b_fc1_critic = self.bias_variable([self.first_fc[1]])

            w_fc2_critic = self.weight_variable(self.second_fc)
            b_fc2_critic = self.bias_variable([self.second_fc[1]])

            w_fc3_critic = self.weight_variable(self.third_fc_critic)
            b_fc3_critic = self.bias_variable([self.third_fc_critic[1]])

        h_fc1_critic = tf.nn.relu(
            tf.matmul(
                self.x,
                w_fc1_critic) +
            b_fc1_critic)
        h_fc2_critic = tf.nn.relu(
            tf.matmul(
                h_fc1_critic,
                w_fc2_critic) +
            b_fc2_critic)

        output_critic = tf.matmul(h_fc2_critic, w_fc3_critic) + b_fc3_critic

        return output_actor, output_critic

    def loss_and_train_critic(self):
        # ----------Critic Loss Function-------------------#
        target_critic = tf.placeholder(tf.float32, shape=[None])
        Loss_critic = tf.reduce_mean(
            tf.square(
                target_critic -
                self.output_critic))
        # Adam
        train_critic = tf.train.AdamOptimizer(
            self.Learning_rate_critic).minimize(Loss_critic)
        # -------------------------------------------#

        return target_critic, train_critic

    def loss_and_train_actor(self):
        # ----------Actor Loss Function-------------------#
        action_actor = tf.placeholder(
            tf.float32, shape=[
                None, self.Num_action])
        advantage_actor = tf.placeholder(tf.float32, shape=[None])
        action_prob = tf.reduce_sum(
            tf.multiply(
                action_actor,
                self.output_actor))
        cross_entropy = tf.multiply(
            tf.log(
                action_prob + 1e-10),
            advantage_actor)
        Loss_actor = - tf.reduce_sum(cross_entropy)
        # Adam
        train_actor = tf.train.AdamOptimizer(
            self.Learning_rate_actor).minimize(Loss_actor)
        # -------------------------------------------#
        return action_actor, advantage_actor, train_actor


if __name__ == '__main__':
    actor_critic = ActorCritic()
    actor_critic.main()
