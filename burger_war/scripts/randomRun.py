#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import rospy

from MyEnv import MyEnv
from MyAgent import MyAgent


backupRoutes = [
    (-0.9, 0.2, 25 / 180.0 * math.pi),    # Pudding_S
    (-0.8, -0.4, -20 / 180.0 * math.pi),  # OctopusWiener_S
    (-0.6, 0.0, 0 / 180.0 * math.pi),     # FriedShrimp_S

    (0.0, 0.5, -180 / 180.0 * math.pi),   # Pudding_N
    (0.0, 0.5, -90 / 180.0 * math.pi),    # FriedShrimp_W
    (0.0, 0.5, 0 / 180.0 * math.pi),      # Tomato_S

    (0.9, 0.25, 115 / 180.0 * math.pi),   # Tomato_N
    (0.55, 0.0, 180 / 180.0 * math.pi),   # FriedShrimp_N
    (0.9, -0.25, -115 / 180.0 * math.pi), # Omelette_N

    (0.0, -0.5, 0 / 180.0 * math.pi),     # Omelette_S
    (0.0, -0.5, 90 / 180.0 * math.pi),    # FriedShrimp_E
    (0.0, -0.5, 180 / 180.0 * math.pi),   # OctopusWiener_N
]

gamma = 0.99
def CreateBatch(agent, replay_memory, batch_size):
    minibatch = random.sample(replay_memory, batch_size)
    state, action, reward, state2, end_flag =  map(np.array, zip(*minibatch))

    x_batch = state
    next_v_values = agent.t_net_v.predict_on_batch(state2)
    y_batch = np.zeros(batch_size)

    for i in range(batch_size):
        y_batch[i] = reward[i] + gamma * next_v_values[i]
    # return [x_batch, action], y_batch
    return [x_batch, action.reshape(batch_size, 3)], y_batch    # TODO: 直値の対応

class RandomBot():
    def __init__(self, bot_name="NoName", side='r', trainingFlag=False):
        self.env = MyEnv(side)
        self.agent = MyAgent(self.env)

        # bot parameter
        self.name = bot_name
        self.side = side
        self.trainingFlag = trainingFlag

        # init status
        self.backupStatus = -1
        # # TODO: 未学習なので常にバックアップ動作
        # if 'r' == self.side:
        #     self.backupStatus = 0
        # else:
        #     self.backupStatus = 6

    def strategy(self):
        fig = None

        while not rospy.is_shutdown():
            n_episode = 300 # 繰り返すエピソード回数
            max_memory = 20000 # リプレイメモリの容量
            # batch_size = 256 # いい感じに収束しないときはバッチサイズいろいろ変えてみて 
            batch_size = 32 # いい感じに収束しないときはバッチサイズいろいろ変えてみて 
            save_counter = 0    # モデル保存用カウンタ

            # max_sigma = 0.99 # 付与するノイズの最大分散値
            max_sigma = 0.10 # 付与するノイズの最大分散値
            sigma = max_sigma

            reduce_sigma = max_sigma / n_episode # 1エピソードで下げる分散値

            # リプレイメモリ
            replay_memory = deque()

            # 自己位置履歴
            myposition_memory = deque(maxlen=3)
            for _ in range(3):
                myposition_memory.append((0.0, 0.0, 0.0))

            # ゲーム再スタート
            for episode in range(n_episode):

                print("episode " + str(episode))
                end_flag = False
                state = self.env.reset()

                sigma -= reduce_sigma
                if sigma < 0:
                    sigma = 0

                while not end_flag:
                    # 自己位置取得
                    tempMyPosition = self.env.getMyPosition()
                    myposition_memory.append(tempMyPosition)
                    tempVar = np.var(np.array(myposition_memory), axis=0)

                    # バックアップモードへの遷移判定
                    print('tempVar : {}'.format(tempVar))
                    if( tempVar[0]<1.0 and tempVar[1]<1.0 and tempVar[2]<100 ):
                        # 動いていないと見做してバックアップモードに遷移
                        self.env.setMode('BACKUP')
                        # TODO: MyAgentからActionとして取得する様にしたい

                    # TODO: 学習が進むまでは強制的にバックアップモードにしておく
                    # if self.env.backupStatus < 0:
                    #     self.env.setMode('BACKUP')

                    # 行動決定
                    if self.env.backupStatus < 0:
                        print('Mode : NORMAL')
                        action = self.agent.getAction(state)
                        # 行動にノイズを付与
                        action += np.random.normal(0, sigma, size=3)
                    else:
                        print('Mode : BACKUP {}'.format(self.env.backupStatus))
                        action = backupRoutes[self.env.backupStatus]
                        action = np.array([action])

                    print(action)
                    state2, reward, end_flag, info = self.env.step(action)

                    
                    # 描画
                    mapList = self.env.getMapList()
                    if None == fig:
                        # 描画準備
                        fig = plt.figure()
                        ax = dict()
                        for axIndex, figName in enumerate(mapList):
                            ax[figName] = fig.add_subplot(2, 4, axIndex+1)
                            plt.title(figName)
                    for figName in mapList:
                        ax[figName].imshow(mapList[figName],interpolation='nearest',vmin=0,vmax=1,cmap='hot')
                    plt.pause(0.001)

                    # 前処理
                    # リプレイメモリに保存
                    print('reward : {}'.format(reward))
                    replay_memory.append([state, action, reward, state2, end_flag])
                    # リプレイメモリが溢れたら前から削除
                    if len(replay_memory) > max_memory:
                        replay_memory.popleft()
                    # リプレイメモリが溜まったら学習
                    if self.trainingFlag:
                        if len(replay_memory) > batch_size*4:
                            x_batch, y_batch = CreateBatch(self.agent, replay_memory, batch_size)
                            self.agent.Train(x_batch, y_batch)

                    # 時々T-NETに重みをコピー
                    if self.trainingFlag:
                        save_counter += 1
                        if save_counter > 600:
                            save_counter = 0
                            # Q-networkの重みをTarget-networkにコピー
                            self.agent.WeightCopy()
                            # ついでにモデルを保存
                            self.agent.saveModel()
                    
                    state = state2

                # 4episodeに1回ターゲットネットワークに重みをコピー
                if episode != 0 and episode % 4 == 0:
                    # Q-networkの重みをTarget-networkにコピー
                    self.agent.WeightCopy()
                    self.agent.t_net_q.save_weights("weight.h5")
                    tempName = 'weight_ep{}.h5'.format(episode)
                    self.agent.t_net_q.save_weights(tempName)


if __name__ == '__main__':
    rospy.init_node('MaruMI')
    side = rospy.get_param('~side', 'r')
    # bot = RandomBot('MaruMI', side, True)
    bot = RandomBot('MaruMI', side, False)
    bot.strategy()

