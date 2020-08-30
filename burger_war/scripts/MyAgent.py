#!/usr/bin/env python
# -*- coding: utf-8 -*-

# renderしなければ動作する
# https://qiita.com/ReoNagai/items/936a2981d6a3e2b000d4

import os
import gym
import random
import math
from operator import mul

import numpy as np
import keras
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model, save_model
from keras.layers import Input, GlobalAveragePooling2D, Concatenate, Lambda
from keras.layers.core import Dense
import tensorflow as tf
from keras.optimizers import Adam

from collections import deque

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

from MyEnv import MyEnv

gamma = 0.99

np.set_printoptions(suppress=True)

def createInceptionv3Module(input, outputFilter):
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell1)
    cell1 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell1)

    cell2 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)
    cell2 = Conv2D(filters=outputFilter//4, kernel_size=3, activation='relu', padding='same')(cell2)

    cell3 = MaxPooling2D(pool_size=2, strides=1, padding='same')(input)
    cell3 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(cell3)

    cell4 = Conv2D(filters=outputFilter//4, kernel_size=1, activation='relu', padding='same')(input)

    output = Concatenate()([cell1, cell2, cell3, cell4])
    return output

class MyAgent:
    def __init__(self, env, ):
        self.env = env
        self.input_dim = env.observation_space.shape
        self.state_dim = env.observation_space.shape
        self.actions_dim = 3

        self.q_net_q, self.q_net_a, self.q_net_v = self.createModel()
        self.t_net_q, self.t_net_action, self.t_net_v = self.createModel()
        self.loadModel()
        # 損失関数
        adam = Adam(lr=0.001, clipnorm=1.)
        # モデル生成
        self.q_net_q.compile(optimizer=adam, loss='mae')
        self.t_net_q.compile(optimizer=adam, loss='mae')

    def createModel(self):
#        x = Input(shape=(self.state_dim,), name='observation_input')
        x = Input(shape=self.state_dim, name='observation_input')
        u = Input(shape=(self.actions_dim,), name='action_input')

        # Middleware
        h = createInceptionv3Module(x, 32)
        h = createInceptionv3Module(h, 64)
        h = createInceptionv3Module(h, 128)
        h = GlobalAveragePooling2D()(h)
        h = Dense(64, activation='relu')(h)
        h = Dense(16, activation='relu')(h)

        # NAF Head
        # < Value function >
        V = Dense(1, activation="linear", name='value_function')(h)

        # < Action Mean >
        mu = Dense(self.actions_dim, activation="tanh", name='action_mean')(h)

        # < L function -> P function >
        l0 = Dense(int(self.actions_dim * (self.actions_dim + 1) / 2), activation="linear", name='l0')(h)
        l1 = Lambda(lambda x: tf.contrib.distributions.fill_triangular(x))(l0)
        L = Lambda(lambda x: tf.matrix_set_diag(x, tf.exp(tf.matrix_diag_part(x))))(l1)
        P = Lambda(lambda x: tf.matmul(x, tf.matrix_transpose(x)))(L)

        # < Action function >
        u_mu = keras.layers.Subtract()([u, mu])
        u_mu_P = keras.layers.Dot(axes=1)([u_mu, P]) # transpose 自動でされてた
        u_mu_P_u_mu = keras.layers.Dot(axes=1)([u_mu_P, u_mu])
        A = Lambda(lambda x: -1.0/2.0 * x)(u_mu_P_u_mu)

        # < Q function >
        Q = keras.layers.Add()([A, V])

        # Input and Output
        model_q = Model(input=[x, u], output=[Q])
        model_mu = Model(input=[x], output=[mu])
        model_v = Model(input=[x], output=[V])
        model_q.summary()
        model_mu.summary()
        model_v.summary()

        return model_q, model_mu, model_v

    def saveModel(self, model_q_path='./model_q.h5', model_a_path='./model_a.h5', model_v_path='./model_v.h5'):
        self.q_net_q.save_weights(model_q_path)
        self.q_net_a.save_weights(model_a_path)
        self.q_net_v.save_weights(model_v_path)
        print('Save : {} {} {}'.format(model_q_path, model_a_path, model_v_path))

    def loadModel(self, model_q_path='./model_q.h5', model_a_path='./model_a.h5', model_v_path='./model_v.h5'):
        if os.path.isfile(model_q_path):
            self.q_net_q.load_weights(model_q_path)
            print('Load : {}'.format(model_q_path))
        if os.path.isfile(model_a_path):
            self.q_net_a.load_weights(model_a_path)
            print('Load : {}'.format(model_a_path))
        if os.path.isfile(model_v_path):
            self.q_net_v.load_weights(model_v_path)
            print('Load : {}'.format(model_v_path))

    def getAction(self, state):
        state = state.reshape((1,) + state.shape)
        action = self.q_net_a.predict_on_batch(state)
        # action = map(mul, action, [1.4, 1.4, math.pi])
        action = action * np.array([[1.4, 1.4, math.pi]])
        return action

    def Train(self, x_batch, y_batch):
        return self.q_net_q.train_on_batch(x_batch, y_batch)

    def PredictT(self, x_batch):
        return self.t_net_q.predict_on_batch(x_batch)

    def WeightCopy(self):
        self.t_net_q.set_weights(self.q_net_q.get_weights())







# def CreateBatch(agent, replay_memory, batch_size):
#     minibatch = random.sample(replay_memory, batch_size)
#     state, action, reward, state2, end_flag =  map(np.array, zip(*minibatch))

#     x_batch = state
#     next_v_values = agent.t_net_v.predict_on_batch(state2)
#     y_batch = np.zeros(batch_size)

#     for i in range(batch_size):
#         y_batch[i] = reward[i] + gamma * next_v_values[i]
#     return [x_batch, action], y_batch



# def main():
#     n_episode = 150 # 繰り返すエピソード回数
#     max_memory = 20000 # リプレイメモリの容量
#     batch_size = 256 # いい感じに収束しないときはバッチサイズいろいろ変えてみて 

#     max_sigma = 0.99 # 付与するノイズの最大分散値
#     sigma = max_sigma

#     reduce_sigma = max_sigma / n_episode # 1エピソードで下げる分散値

#     # env = gym.make("Pendulum-v0") # 環境
#     env = MyEnv('r')    # TODO: side
#     agent = MyAgent(env)
#     # リプレイメモリ
#     replay_memory = deque()

#     # ゲーム再スタート
#     for episode in range(n_episode):

#         print("episode " + str(episode))
#         end_flag = False
#         state = env.reset()

#         sigma -= reduce_sigma
#         if sigma < 0:
#             sigma = 0

#         while not end_flag:
#             # 行動にノイズを付与
#             action = agent.getAction(state) + np.random.normal(0, sigma, size=3)
#             # action = action[:,0]
#             # # Pendulumの行動が-2~2の範囲なので変換
#             # action = np.clip(action, -1.0, 1.0) * 2.0

#             state2, reward, end_flag, info = env.step(action)
#             # 前処理
#             # リプレイメモリに保存
#             replay_memory.append([state, action, reward, state2, end_flag])
#             # リプレイメモリが溢れたら前から削除
#             if len(replay_memory) > max_memory:
#                 replay_memory.popleft()
#             # リプレイメモリが溜まったら学習
#             if len(replay_memory) > batch_size*4:
#                 x_batch, y_batch = CreateBatch(agent, replay_memory, batch_size)
#                 agent.Train(x_batch, y_batch)

#             state = state2
#             # 可視化をする場合はこのコメントアウトを解除
#             # env.render()

#         # 4episodeに1回ターゲットネットワークに重みをコピー
#         if episode != 0 and episode % 4 == 0:
#             agent.WeightCopy()
#             # Q-networkの重みをTarget-networkにコピー
#             agent.t_net_q.save_weights("weight.h5")

#     env.close()

#     agent.WeightCopy()
#     # Q-networkの重みをTarget-networkにコピー
#     agent.t_net_q.save_weights("weight.h5")


# if __name__ == '__main__':
#     main()
