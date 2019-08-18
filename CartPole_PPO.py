#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------
# Author: WangCai
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

env = gym.make('CartPole-v0')
env.seed(0)

# Choose the method for optimization
METHOD = [
    dict(name='KLPEN', kl_target = 0.01, beta = 0.5),
    dict(name='CLIP', epsilon = 0.2),
    dict(name='CLIP_VF_S', epsilon = 0.2, c_1 = 1.0, c_2 = 0.01, Learning_rate = 0.005)][2]
# The adaptive KL penatly is described in eq (8),
# and the importance ratio clipping is described in eq (7),
# and the simplified surrogate loss is described in eq (9)
# of "Proximal Policy Optimization Algorithms".
# Note that the c_1 is the parameter for value difference and c_2 is parameter for entropy bonus.
print("OPTIMAZATION METHOD ---------- {}".format(METHOD['name']))

class PPO:
    def __init__(self):
        self.algorithm = 'PPO'
        self.game_name = 'CartPole'

        # Input
        self.observation = env.observation_space
        # Action size
        self.Num_action = env.action_space.n   

        self.Num_plot_episode = 1

        self.Gamma = 0.98
        
        # The parameter of generative advantage estimator in eq (11)
        self.Lambda = 1

        # Date of training
        self.date_time = str(datetime.date.today())

        self.Is_solved = False

        # Critic network parameters
        self.critic_first_fc = list(self.observation.shape) + [100]
        self.critic_second_fc = [100, 1]
        # Actor network parameters
        self.actor_first_fc = list(self.observation.shape)+[40]
        self.actor_second_fc = [40, 35]
        self.actor_third_fc = [35, 30]
        self.actor_fourth_fc = [30, self.Num_action]

        # Learning rate
        self.Learning_rate_actor = 0.01
        self.Learning_rate_critic = 0.001
        self.critic_update_steps = 1
        self.actor_update_steps = 5

        # Initialization
        self.act_prob_pi, self.act_prob_old_pi, self.output_critic = self.network()

        # -----------Loss and train-------------
        self.train_critic, self.rollout, self.loss_critic = self.loss_and_train_critic()
        if METHOD['name'] == 'KLPEN':
            self.beta, self.kl_mean, self.chosen_actions, \
                self.advantage, self.train_actor = self.loss_and_train_actor()
        elif METHOD['name'] == 'CLIP':
            self.chosen_actions, self.advantage, self.train_actor = self.loss_and_train_actor()
        elif METHOD['name'] == 'CLIP_VF_S':
            self.chosen_actions, self.advantage, self.train = self.loss_and_train_actor()
        # -----------------------------------

        self.sess, self.saver = self.init_sess()

    def main(self):
        env = gym.make('CartPole-v0')

        # Plot
        plt.figure(1)
        plot_x = []
        plot_y = []

        episode = 1
        running_reward = []
        step = 0
        while(True):
            state = env.reset()
            terminal = False
            ep_rewards = []  # Episode' rewards
            ep_actions = []  # Episode' actions
            ep_states = []  # Episode' states
            score = 0
            while not terminal:
                env.render()  # Render the image of the cartpole
                action = ppo.choose_action(state)
                a_binarized = np.zeros(self.Num_action)
                a_binarized[action] = 1
                state_next, reward, terminal, _ = env.step(action)
                score += reward
                ep_actions.append(a_binarized)
                ep_rewards.append(reward)
                ep_states.append(state)
                step += 1
                state = state_next
                if terminal:
                    # End of an episode
                    plot_x.append(episode)
                    plot_y.append(score)
                    ep_actions = np.vstack(ep_actions)
                    ep_rewards = np.array(ep_rewards, dtype=np.float_)
                    ep_states = np.vstack(ep_states)

                    # Estimate advantages
                    ep_advantages = self.calculate_advantages(
                        ep_rewards, self.get_v(ep_states))

                    targets = self.discount_rewards(ep_rewards)

                    # update actor_oldpi network
                    self.assign_pi_network_to_oldpi_network()

                    # -----------Update actor and critic network-------------
                    if METHOD['name'] == 'KLPEN':
                        # Update actor network
                        for _ in range(self.actor_update_steps):
                            _, kl = self.sess.run(
                                [self.train_actor, self.kl_mean],
                                feed_dict={
                                    self.x: ep_states,
                                    self.chosen_actions: ep_actions,
                                    self.advantage: ep_advantages,
                                    self.beta: METHOD['beta']
                                }
                            )
                            # print("kl={}".format(kl))
                            if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                                break
                        # adaptive beta, this is in OpenAI's paper
                        if kl < METHOD['kl_target'] / 1.5:
                            METHOD['beta'] /= 2
                        elif kl > METHOD['kl_target'] * 1.5:
                            METHOD['beta'] *= 2
                        # sometimes explode
                        METHOD['beta'] = np.clip(METHOD['beta'], 1e-4, 10)
                        # Update critic network
                        for _ in range(self.critic_update_steps):
                            self.train_critic.run(
                                feed_dict={
                                    self.rollout: targets,
                                    self.x: ep_states})
                    elif METHOD['name'] == 'CLIP':
                        # Update actor network
                        for _ in range(self.actor_update_steps):
                            self.train_actor.run(
                                feed_dict={
                                    self.x: ep_states,
                                    self.chosen_actions: ep_actions,
                                    self.advantage: ep_advantages})
                        # Update critic network
                        for _ in range(self.critic_update_steps):
                            self.train_critic.run(
                                feed_dict={
                                    self.rollout: targets,
                                    self.x: ep_states})
                    elif METHOD['name'] == 'CLIP_VF_S':
                        self.train.run(
                            feed_dict={
                                self.x:ep_states,
                                self.chosen_actions:ep_actions,
                                self.advantage:ep_advantages,
                                self.rollout:targets,})                        
                    # -----------------------------------

                    ep_rewards = []
                    ep_actions = []
                    ep_states = []
                    running_reward.append(score)
                    if episode % self.Num_plot_episode == 0:
                        avg_score = np.mean(running_reward[-25:])
                        print('step:' + str(step) + '/' +
                              'episode:' + str(episode) + '/' +
                              'score:' + str(score))
                        if avg_score >= 200:
                            print("Solved!")
                            self.Is_solved = True
                            plt.savefig(
                                self.date_time + '_' +
                                self.algorithm + '_' +
                                self.game_name + '.png')
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

                        # Clear
                        plot_x = []
                        plot_y = []

                    episode += 1

            if self.Is_solved:
                break

    def discount_rewards(self, rewards):
        running_total = 0
        discounted = np.zeros_like(rewards)
        for r in reversed(range(len(rewards))):
            running_total = running_total * self.Gamma + rewards[r]
            discounted[r] = running_total
        return discounted

    def calculate_advantages(self, rewards, values):
        advantages = np.zeros_like(rewards)
        # calculate generative advantage estimator(Lambda = self.Lambda), see ppo paper eq(11)
        for t in range(len(rewards)):
            ad = 0
            for l in range(0, len(rewards) - t - 1):
                delta = rewards[t + l] + self.Gamma * \
                    values[t + l + 1] - values[t + l]
                ad += ((self.Gamma * self.Lambda)**l) * (delta)
            ad += ((self.Gamma * self.Lambda)**l) * \
                (rewards[t + l] - values[t + l])
            advantages[t] = ad
        return (advantages - np.mean(advantages)) / np.std(advantages)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # get probabilities for all actions
        probs = self.sess.run(self.act_prob_pi, {self.x: s})
        # print("probs={}".format(probs))
        action_step = np.random.choice(
            np.arange(probs.shape[1]), p=probs.ravel())

        return np.clip(action_step, -2, 2)

    def init_sess(self):
        config = tf.ConfigProto()
        sess = tf.InteractiveSession(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        return sess, saver

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

    def network(self):
        tf.reset_default_graph()

        self.x = tf.placeholder(tf.float32, shape=[None]+list(self.observation.shape))

        with tf.variable_scope('Actor_pi'):
            w_fc1_actor_pi = self.weight_variable('_w_fc1', self.actor_first_fc)
            b_fc1_actor_pi = self.bias_variable('_b_fc1', [self.actor_first_fc[1]])

            w_fc2_actor_pi = self.weight_variable('_w_fc2', self.actor_second_fc)
            b_fc2_actor_pi = self.bias_variable('_b_fc2', [self.actor_second_fc[1]])

            w_fc3_actor_pi = self.weight_variable('_w_fc3', self.actor_third_fc)
            b_fc3_actor_pi = self.bias_variable('_b_fc3', [self.actor_third_fc[1]])

            w_fc4_actor_pi = self.weight_variable('_w_fc4', self.actor_fourth_fc)
            b_fc4_actor_pi = self.bias_variable('_b_fc4', [self.actor_fourth_fc[1]])

        h_fc1_actor_pi = tf.nn.tanh(tf.matmul(self.x, w_fc1_actor_pi) + b_fc1_actor_pi)
        h_fc2_actor_pi = tf.nn.tanh(tf.matmul(h_fc1_actor_pi, w_fc2_actor_pi) + b_fc2_actor_pi)
        h_fc3_actor_pi = tf.nn.tanh(tf.matmul(h_fc2_actor_pi, w_fc3_actor_pi) + b_fc3_actor_pi)
        act_prob_pi = tf.nn.softmax(tf.matmul(h_fc3_actor_pi, w_fc4_actor_pi) + b_fc4_actor_pi)

        with tf.variable_scope('Actor_old_pi'):
            w_fc1_actor_old_pi = self.weight_variable('_w_fc1', self.actor_first_fc)
            b_fc1_actor_old_pi = self.bias_variable('_b_fc1', [self.actor_first_fc[1]])

            w_fc2_actor_old_pi = self.weight_variable('_w_fc2', self.actor_second_fc)
            b_fc2_actor_old_pi = self.bias_variable('_b_fc2', [self.actor_second_fc[1]])

            w_fc3_actor_old_pi = self.weight_variable('_w_fc3', self.actor_third_fc)
            b_fc3_actor_old_pi = self.bias_variable('_b_fc3', [self.actor_third_fc[1]])

            w_fc4_actor_old_pi = self.weight_variable('_w_fc4', self.actor_fourth_fc)
            b_fc4_actor_old_pi = self.bias_variable('_b_fc4', [self.actor_fourth_fc[1]])

        h_fc1_actor_old_pi = tf.nn.tanh(tf.matmul(self.x,w_fc1_actor_old_pi) +b_fc1_actor_old_pi)
        h_fc2_actor_old_pi = tf.nn.tanh(tf.matmul(h_fc1_actor_old_pi,w_fc2_actor_old_pi) + b_fc2_actor_old_pi)
        h_fc3_actor_old_pi = tf.nn.tanh(tf.matmul(h_fc2_actor_old_pi,w_fc3_actor_old_pi) + b_fc3_actor_old_pi)
        act_prob_old_pi = tf.nn.softmax(tf.matmul(h_fc3_actor_old_pi, w_fc4_actor_old_pi) + b_fc4_actor_old_pi)

        with tf.variable_scope('critic'):
            w_fc1_critic = self.weight_variable('_w_fc1', self.critic_first_fc)
            b_fc1_critic = self.bias_variable('_b_fc1', [self.critic_first_fc[1]])

            w_fc2_critic = self.weight_variable('_w_fc2', self.critic_second_fc)
            b_fc2_critic = self.bias_variable('_b_fc2', [self.critic_second_fc[1]])

        h_fc1_critic = tf.nn.tanh(tf.matmul(self.x, w_fc1_critic) + b_fc1_critic)
        output_critic = tf.matmul(h_fc1_critic, w_fc2_critic) + b_fc2_critic

        return act_prob_pi, act_prob_old_pi, output_critic

    def loss_and_train_critic(self):
        rollout = tf.placeholder(
            tf.float32, shape=[None])  # discounted_rewards
        loss_critic = tf.reduce_mean(tf.square(rollout - self.output_critic))
        train_critic = tf.train.AdamOptimizer(
            self.Learning_rate_critic).minimize(loss_critic)

        return train_critic, rollout, loss_critic

    def loss_and_train_actor(self):
        chosen_actions = tf.placeholder(
            tf.float32, shape=[None, self.Num_action])  # one-hot

        advantage = tf.placeholder(tf.float32, shape=[None])

        new_responsible_outputs = tf.reduce_sum(chosen_actions * self.act_prob_pi, axis=1)
        old_responsible_outputs = tf.reduce_sum(chosen_actions * self.act_prob_old_pi, axis=1)

        # ratio = new_responsible_outputs / old_responsible_outputs
        ratio = tf.exp(tf.log(new_responsible_outputs) - tf.log(old_responsible_outputs))

        surr = ratio * advantage
        
        if METHOD['name'] == 'KLPEN':
            beta = tf.placeholder(tf.float32, None, 'beta')
            kl = tf.reduce_sum(
                tf.multiply(
                    self.act_prob_old_pi,
                    tf.log(
                        tf.div(
                            self.act_prob_old_pi,
                            self.act_prob_pi))), axis=1)
            kl_mean = tf.reduce_mean(kl)
            loss_actor = -tf.reduce_mean(surr - beta * kl)
            train_actor = tf.train.AdamOptimizer(
                self.Learning_rate_actor).minimize(loss_actor)
            return beta, kl_mean, chosen_actions, advantage, train_actor
        elif METHOD['name'] == 'CLIP':
            clip_value = tf.clip_by_value(
                ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon'])
            loss_actor = - \
                tf.reduce_mean(tf.minimum(surr, clip_value * advantage))
            train_actor = tf.train.AdamOptimizer(
                self.Learning_rate_actor).minimize(loss_actor)
            return chosen_actions, advantage, train_actor
        elif METHOD['name'] == 'CLIP_VF_S':
            clip_ratio= tf.clip_by_value(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon'])
            loss_clip = tf.reduce_mean(tf.minimum(surr, clip_ratio * advantage))
            clip_new_act_prob_pi = tf.clip_by_value(self.act_prob_pi, 1e-10, 1.0)
            entropy = -tf.reduce_sum(self.act_prob_pi*tf.log(clip_new_act_prob_pi), axis=1)
            entropy = tf.reduce_mean(entropy)
            loss_CLIP_VF_S = -(loss_clip- METHOD['c_1'] * self.loss_critic + METHOD['c_2'] * entropy)
            train = tf.train.AdamOptimizer(METHOD['Learning_rate']).minimize(loss_CLIP_VF_S)
            return chosen_actions, advantage, train

    def get_v(self, s):
        # if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.output_critic, {self.x: s})

    def assign_pi_network_to_oldpi_network(self):
        trainable_variables = tf.trainable_variables()
        trainable_variables_old_pi_network = [
            var for var in trainable_variables
            if var.name.startswith('Actor_old_pi')]
        trainable_variables_pi_network = [
            var for var in trainable_variables
            if var.name.startswith('Actor_pi')]

        for i in range(len(trainable_variables_old_pi_network)):
            self.sess.run(
                tf.assign(
                    trainable_variables_old_pi_network[i],
                    trainable_variables_pi_network[i]))

if __name__ == '__main__':
    ppo = PPO()
    ppo.main()
