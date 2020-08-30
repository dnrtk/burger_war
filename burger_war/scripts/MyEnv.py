#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import threading
import time
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.spaces

import rospy
import actionlib
import tf
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

FIELD_SIZE = (80, 80)

def euler_to_quaternion(euler):
    q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def goal_pose(targetX, targetY, targetR):
    # targetX : -1.4 〜 1.4
    # targetY : -1.4 〜 1.4
    # targetR : -3.14 〜 3.14 [rad]
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'

    goal_pose.target_pose.pose.position.x = targetX
    goal_pose.target_pose.pose.position.y = targetY
    goal_pose.target_pose.pose.position.z = 0.0

    # targetR = targetR / 180.0 * math.pi
    tempQuaternion = euler_to_quaternion(Vector3(0.0, 0.0, targetR))
    goal_pose.target_pose.pose.orientation = tempQuaternion
    
    return goal_pose

def occupancyGrid_to_ndarray(occupancyGridData, visibleFlag=False):
    # [0 〜1.0] に正規化
    # print('height:{} width:{}'.format(occupancyGridData.info.height, occupancyGridData.info.width))
    # print('min:{} max:{}'.format(min(occupancyGridData.data), max(occupancyGridData.data)))
    tempMapData = np.array(occupancyGridData.data)
    tempMapData -= min(tempMapData)
    tempMapData /= max(tempMapData)
    tempMapData = tempMapData.reshape((occupancyGridData.info.height, occupancyGridData.info.width))
    # 不要な領域が多いので80x80にトリミングする
    tempMapData = tempMapData[160:240, 160:240]

    # 確認用にマップ可視化
    if visibleFlag:
        plt.figure()
        plt.imshow(tempMapData,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
        plt.colorbar()
        plt.show()
        # plt.pause(0.001)

    return tempMapData


# 参考
# https://qiita.com/ohtaman/items/edcb3b0a2ff9d48a7def
# https://qiita.com/inoory/items/e63ade6f21766c7c2393


class MyEnv(gym.Env):
    mapScaleMax = 100
    def __init__(self, side, renderFlag=False):
        # super().__init__()
        self.side = side
        self.renderFlag = renderFlag
        self.stepCount = 0
        self.preReward = 0
        self.backupStatus = -1
        # 環境定義
        self.action_space = gym.spaces.Discrete(3) # X座標, Y座標, 方向 の3次元
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(80, 80, 7), dtype=np.float32) # 観測空間(state)の次元とそれらの最大値
        self.reward_range = [-14., 14.] # 最大の得失点差が14
        # Navigation準備
        self.r = rospy.Rate(1) # change speed 1fps
        self.listener = tf.TransformListener()
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.listener.waitForTransform("map", "base_link", rospy.Time(), rospy.Duration(4.0))
        # Map初期化
        self.initMapList()
        # TOPIC受信準備
        self.setTopicSubscriber()
        # self._reset()
        # 描画準備
        if self.renderFlag:
            self.fig = plt.figure()
            self.ax = dict()
            for axIndex, figName in enumerate(self.mapList):
                self.ax[figName] = self.fig.add_subplot(2, 4, axIndex+1)
                plt.title(figName)

    def initMapList(self):
        # initialize map channel
        self.mapList = dict()
        self.mapList['FieldMap'] = np.zeros(FIELD_SIZE)
        self.mapList['LidarMap'] = np.zeros(FIELD_SIZE)
        self.mapList['MyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyMap'] = np.zeros(FIELD_SIZE)
        self.mapList['MyPositionMap'] = np.zeros(FIELD_SIZE)

    def setTopicSubscriber(self):
        self.subscribers = dict()
        self.subscribers['map'] = MapTopic()
        self.subscribers['war_state'] = WarStateTopic(self.side)
        self.subscribers['odom'] = OdometryTopic()
        # self.gCostMap_sub = rospy.Subscriber('move_base/global_costmap/costmap', OccupancyGrid, self.globalCostmapCallback)

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def step(self, action):
        self.stepCount += 1

        # # 入力用に加工
        # inputImage = np.array(list(self.mapList.values()))
        # inputImage = inputImage.reshape(1, 80, 80, 7)
        # # inputImage *= 100
        # # 行動決定
        # tempPose = self.myModel.predict(inputImage)
        tempPose = action[0]
        tempGoal = goal_pose(tempPose[0], tempPose[1], tempPose[2])

        print('Goal: {} {} {}'.format(tempPose[0], tempPose[1], tempPose[2]))
        self.client.send_goal(tempGoal, done_cb=self.done_cb, feedback_cb=self.feedback_cb)
        # self.client.send_goal(tempGoal)
        self.r.sleep()
        # wait_for_result(self, timeout = rospy.Duration()) でNavigationのgoal到達までを待てる

        # state2, reward, end_flag, info = env.step(action)
        state2 = self.observe()
        nowReward = self.subscribers['war_state'].getReward()
        reward = nowReward - self.preReward
        self.preReward = nowReward
        end_flag = False if self.stepCount < 180 else True  # 180sec経過で終了
        end_flag = False    # TODO: Resetに失敗するので、無限に続行
        info = {}
        return state2, reward, end_flag, info


    def getMyPosition(self):
        return self.subscribers['odom'].getMyPosition()

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def reset(self):
        print('MyEnv Reset')
        # TODO: 環境リセット
        self.stepCount = 0
        self.preReward = 0
        time.sleep(0.5)
        return self.observe()

    # 現在保持しているtopicから環境を生成
    def observe(self):
        self.mapList['FieldMap'] = self.subscribers['map'].getMap()
        # self.mapList['LidarMap'] = 
        # self.mapList['MyQrMap'], self.mapList['EnemyQrMap'], self.mapList['NoneQrMap'] = self.subscribers['war_state'].getMap()
        self.subscribers['war_state'].getMap()
        self.mapList['MyQrMap'] = self.subscribers['war_state'].mapList['MyQrMap']
        self.mapList['EnemyQrMap'] = self.subscribers['war_state'].mapList['EnemyQrMap']
        self.mapList['NoneQrMap'] = self.subscribers['war_state'].mapList['NoneQrMap']
        # self.mapList['EnemyMap'] = 
        self.mapList['MyPositionMap'] = self.subscribers['odom'].getMap()
        # return self.mapList
        # return list(self.mapList.values())
        return np.array(list(self.mapList.values())).transpose(1,2,0) * MyEnv.mapScaleMax

    def feedback_cb(self,feed):
        # print('feedback_cb: {}'.format(feed))
        pass

    def done_cb(self,result, result2):
        # print('done_cb: {}'.format(result))
        if self.backupStatus >= 0:
            self.backupStatus += 1
            # if self.backupStatus >= len(backupRoutes):
            if self.backupStatus >= 12:
                # 一巡したらバックアップモード解除
                self.backupStatus = -1
                print('Mode Change : NORMAL')
            print('BackupStatus:{}'.format(self.backupStatus))

    def setMode(self, mode):
        if 'BACKUP' == mode:
            self.backupStatus = 0
            print('Mode Change : BACKUP')
        else:
            self.backupStatus = -1
            print('Mode Change : NORMAL')

    # def showMap(self):
    #     if self.renderFlag:
    #         for figName in self.mapList:
    #             self.ax[figName].imshow(self.mapList[figName],interpolation='nearest',vmin=0,vmax=1,cmap='hot')
    #         plt.pause(0.001)

    def getMapList(self):
        return self.mapList


class MapTopic:
    def __init__(self):
        self.mapList = dict()
        self.mapList['FieldMap'] =  np.zeros(FIELD_SIZE)

        self.lock = threading.RLock()
        self.subscriber = rospy.Subscriber('map', OccupancyGrid, self._mapCallback)

    def _mapCallback(self, data):
        # print('MapCB')
        with self.lock:
            self.map = data

    def getMap(self):
        with self.lock:
            self.mapList['FieldMap'] = occupancyGrid_to_ndarray(self.map)
        return self.mapList['FieldMap']


class WarStateTopic:
    qrCoordinates = {
        'Tomato_N' : (27, 29),
        'Tomato_S' : (32, 29),
        'Omelette_N' : (27, 50),
        'Omelette_S' : (32, 50),
        'Pudding_N' : (48, 29),
        'Pudding_S' : (53, 29),
        'OctopusWiener_N' : (48, 50),
        'OctopusWiener_S' : (53,50),
        'FriedShrimp_N' : (36, 40),
        'FriedShrimp_E' : (40, 44),
        'FriedShrimp_W' : (40, 35),
        'FriedShrimp_S' : (45, 40),
    }

    def __init__(self, side):
        self.side = side
        self.mapList = dict()
        self.mapList['MyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] = np.zeros(FIELD_SIZE)

        self.lock = threading.RLock()
        self.subscriber = rospy.Subscriber('war_state', String, self._warStateCallback)

    def _warStateCallback(self, data):
        # print('WarStateCB')
        with self.lock:
            self.warState = data.data
        
    def getMap(self):
        self.mapList['MyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] = np.zeros(FIELD_SIZE)

        try:
            with self.lock:
                warStateDict = json.loads(self.warState)
        except:
            # パース失敗時は空MAPにする
            return
        # print(warStateDict['players'])
        # print(warStateDict['scores'])
        for target in warStateDict['targets']:
            # print(target)
            if target['name'] in WarStateTopic.qrCoordinates:
                tempMapName = 'EnemyQrMap'
                if 'n' == target['player']:
                    tempMapName = 'NoneQrMap'
                elif self.side == target['player']:
                    tempMapName = 'MyQrMap'
                tempQrCoordinates = WarStateTopic.qrCoordinates[target['name']]
                self.mapList[tempMapName][tempQrCoordinates[0]][tempQrCoordinates[1]] = 0.2 * int(target['point'])  #フィールドのマーカーは1pointなので0.2
        return self.mapList['MyQrMap'], self.mapList['EnemyQrMap'], self.mapList['NoneQrMap']
    
    def getScore(self):
        try:
            with self.lock:
                warStateDict = json.loads(self.warState)
        except:
            # パース失敗時は0ポイントを返す
            print('getScore() : Parse Error')
            return {'r':0, 'b':0}
        return warStateDict['scores']
    
    def getReward(self):
        tempScore = self.getScore()
        enemySide = 'b' if 'r' == self.side else 'r'
        try:
            tempReward = int(tempScore[self.side]) - int(tempScore[enemySide])
        except:
            print('getReward() : Get Score Failed')
            tempReward = 0
        return tempReward

class OdometryTopic:
    myRobotMap = np.array([
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0.5, 0.5, 0.5,	1,	1,	1,	1,	1,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0.5, 0.5, 0.5,	1,	1,	1,	1,	1,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0.5, 0.5, 0.5,	1,	1,	1,	1,	1,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
        [0,	0,	0,	0,	0,	0,	0,	0,	 0,	  0,	0,	0,	0,	0,	0,	0,	0],
    ])
    myRobotMapCenterY = int(myRobotMap.shape[0]/2)
    myRobotMapCenterX = int(myRobotMap.shape[1]/2)

    def __init__(self):
        self.mapList = dict()
        self.mapList['MyPositionMap'] =  np.zeros(FIELD_SIZE)

        self.myX, self.myY, self.myR = 0.0, 0.0, 0.0

        self.lock = threading.RLock()
        self.subscriber = rospy.Subscriber('odom', Odometry, self._odometryCallback)

    def _odometryCallback(self, data):
        # print('OdometryCB')
        with self.lock:
            self.odom = data

    def getMap(self):
        with self.lock:
            # 座標を -2.0〜2.0 → 0.0〜4.0 に変換して、更にフィールドの座標に変換
            self.myX = int((self.odom.pose.pose.position.x + 2.0) / 4.0 * FIELD_SIZE[1])
            self.myY = int(FIELD_SIZE[0] - (self.odom.pose.pose.position.y + 2.0) / 4.0 * FIELD_SIZE[0])
            tempVector3 = quaternion_to_euler(self.odom.pose.pose.orientation)
        self.myR = tempVector3.z / math.pi * 180
        print('x:{} y:{} r:{}'.format(self.myX, self.myY, self.myR))
        
        trans = cv2.getRotationMatrix2D((OdometryTopic.myRobotMapCenterX, OdometryTopic.myRobotMapCenterY), self.myR , 1.0)
        tempMyRobotMap = cv2.warpAffine(OdometryTopic.myRobotMap, trans, OdometryTopic.myRobotMap.shape)

        # 自ロボ位置と射程を描画
        self.mapList['MyPositionMap'] = np.zeros(FIELD_SIZE)
        # self.mapList['MyPositionMap'] += self.mapList['FieldMap']
        offsetX = self.myX - OdometryTopic.myRobotMapCenterX
        offsetY = self.myY - OdometryTopic.myRobotMapCenterY
        for indexY, valueY in enumerate(tempMyRobotMap):
            for indexX, valueX in enumerate(valueY):
                if (indexY + offsetY) < FIELD_SIZE[0] and (indexX + offsetX) < FIELD_SIZE[1]:
                    self.mapList['MyPositionMap'][indexY + offsetY][indexX + offsetX] += valueX
        return self.mapList['MyPositionMap']

    def getMyPosition(self):
        return (self.myX, self.myY, self.myR) 

