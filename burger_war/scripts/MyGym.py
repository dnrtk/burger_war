# -*- coding: utf-8 -*-

import gym
import numpy as np

# 参考
# https://qiita.com/inoory/items/e63ade6f21766c7c2393

class MyGym(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(3) # X座標, Y座標, 方向 の3次元
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(80, 80, 7), dtype=np.float32) # 観測空間(state)の次元とそれらの最大値

    def _reset(self):
        return self._get_observation()

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        pass

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):
        pass

MyGym()
