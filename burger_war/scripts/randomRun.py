# -*- coding: utf-8 -*-

import sys
import random
import json
import math
import threading

import cv2
import numpy as np
import matplotlib.pyplot as plt

import rospy
import actionlib
import tf
from geometry_msgs.msg import Twist
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

from MyModel import MyModel


FIELD_SIZE = (80, 80)

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

# 1Point=0.20, 3Point=0.60, 5Point=1.00
# 1Point=0.20
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

def euler_to_quaternion(euler):
    q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def goal_pose(targetX, targetY, targetR):
    # targetX : -1.4 〜 1.4
    # targetY : -1.4 〜 1.4
    # targetR : -180.0 〜 180.0 [deg]
    goal_pose = MoveBaseGoal()
    goal_pose.target_pose.header.frame_id = 'map'

    goal_pose.target_pose.pose.position.x = targetX
    goal_pose.target_pose.pose.position.y = targetY
    goal_pose.target_pose.pose.position.z = 0.0

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
    # TODO: 不要な領域が多いので80x80位にトリミングする (odom情報との整合性が取れるか確認してから)
    tempMapData = tempMapData[160:240, 160:240]

    # 確認用にマップ可視化
    if visibleFlag:
        plt.figure()
        plt.imshow(tempMapData,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
        plt.colorbar()
        plt.show()
        # plt.pause(0.001)

    return tempMapData

class RandomBot():
    def __init__(self, bot_name="NoName"):
        # bot name 
        self.name = bot_name
        # initialize map channel
        self.mapList = dict()
        self.mapList['FieldMap'] = np.zeros(FIELD_SIZE)
        self.mapList['MyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] = np.zeros(FIELD_SIZE)
        self.mapList['EnemyMap'] = np.zeros(FIELD_SIZE)
        self.mapList['MyPositionMap'] = np.zeros(FIELD_SIZE)

        self.map_data = np.zeros(FIELD_SIZE)
        self.myPositionMap = np.zeros(FIELD_SIZE)
        # velocity publisher
        self.vel_pub = rospy.Publisher('cmd_vel', Twist,queue_size=1)
        self.map_sub = rospy.Subscriber('map', OccupancyGrid, self.mapCallback)
        # self.gCostMap_sub = rospy.Subscriber('move_base/global_costmap/costmap', OccupancyGrid, self.globalCostmapCallback)
        self.warState_lock = threading.RLock()
        self.warState_sub = rospy.Subscriber('war_state', String, self.warStateCallback)
        self.odom = Odometry()
        self.odom_lock = threading.RLock()
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odometryCallback)
    
    def mapCallback(self, data):
        # print('MapCB')
        self.map = data
        # self.map_data = occupancyGrid_to_ndarray(data)
        self.mapList['FieldMap'] = occupancyGrid_to_ndarray(data)

    def globalCostmapCallback(self, data):
        # print('GlobalCostmapCB')
        self.gCostMap = data
        self.gCostMap_data = occupancyGrid_to_ndarray(data)

    def warStateCallback(self, data):
        # print('WarStateCB')
        with self.warState_lock:
            self.warState = data.data
        
    def createQrMap(self):
        self.mapList['MyQrMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] =  np.zeros(FIELD_SIZE)

        try:
            with self.warState_lock:
                warStateDict = json.loads(self.warState)
        except:
            # パース失敗時は空MAPにする
            return
        # print(warStateDict['players'])
        # print(warStateDict['scores'])
        for target in warStateDict['targets']:
            # print(target)
            if target['name'] in qrCoordinates:
                tempMapName = 'EnemyQrMap'
                if 'n' == target['player']:
                    tempMapName = 'NoneQrMap'
                elif 'r' == target['player']:
                    tempMapName = 'MyQrMap'
                tempQrCoordinates = qrCoordinates[target['name']]
                self.mapList[tempMapName][tempQrCoordinates[0]][tempQrCoordinates[1]] = 0.2 * int(target['point'])
        # print(int(warStateDict['targets'][0]['point']))

    def odometryCallback(self, data):
        # print('OdometryCB')
        with self.odom_lock:
            self.odom = data

    def createMyPositionMap(self):
        with self.odom_lock:
            # 座標を -2.0〜2.0 → 0.0〜4.0 に変換して、更にフィールドの座標に変換
            tempX = int((self.odom.pose.pose.position.x + 2.0) / 4.0 * FIELD_SIZE[1])
            tempY = int(FIELD_SIZE[0] - (self.odom.pose.pose.position.y + 2.0) / 4.0 * FIELD_SIZE[0])
            tempVector3 = quaternion_to_euler(self.odom.pose.pose.orientation)
        tempRotate = tempVector3.z / math.pi * 180
        # print('x:{} y:{} r:'.format(tempX, tempY, tempRotate))
        
        trans = cv2.getRotationMatrix2D((myRobotMapCenterX, myRobotMapCenterY), tempRotate , 1.0)
        tempMyRobotMap = cv2.warpAffine(myRobotMap, trans, myRobotMap.shape)

        # 自ロボ位置と射程を描画
        self.mapList['MyPositionMap'] = np.zeros(FIELD_SIZE)
        # self.mapList['MyPositionMap'] += self.mapList['FieldMap']
        offsetX = tempX - myRobotMapCenterX
        offsetY = tempY - myRobotMapCenterY
        for indexY, valueY in enumerate(tempMyRobotMap):
            for indexX, valueX in enumerate(valueY):
                if (indexY + offsetY) < FIELD_SIZE[0] and (indexX + offsetX) < FIELD_SIZE[1]:
                    self.mapList['MyPositionMap'][indexY + offsetY][indexX + offsetX] += valueX

    # def calcTwist(self):
    #     value = random.randint(1,1000)
    #     # value = 1001
    #     if value < 250:
    #         x = 0.2
    #         th = 0
    #     elif value < 500:
    #         x = -0.2
    #         th = 0
    #     elif value < 750:
    #         x = 0
    #         th = 1
    #     elif value < 1000:
    #         x = 0
    #         th = -1
    #     else:
    #         x = 0
    #         th = 0
    #     twist = Twist()
    #     twist.linear.x = x; twist.linear.y = 0; twist.linear.z = 0
    #     twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th
    #     return twist

    def strategy(self):
        r = rospy.Rate(1) # change speed 1fps

        # モデル準備
        self.myModel = MyModel()

        # Navigation準備
        self.listener = tf.TransformListener()
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.listener.waitForTransform("map", "base_link", rospy.Time(), rospy.Duration(4.0))

        # 描画準備
        fig = plt.figure()
        ax = dict()
        for axIndex, figName in enumerate(self.mapList):
            ax[figName] = fig.add_subplot(2, 3, axIndex+1)
            plt.title(figName)

        while not rospy.is_shutdown():
            # 各チャネル情報生成
            self.createMyPositionMap()
            self.createQrMap()

            # 描画
            for figName in self.mapList:
                ax[figName].imshow(self.mapList[figName],interpolation='nearest',vmin=0,vmax=1,cmap='hot')
            plt.pause(0.001)

            # 入力用に加工
            inputImage = np.array(list(self.mapList.values()))
            inputImage = inputImage.reshape(1, 80, 80, 6)
            inputImage *= 100
            
            # 行動決定
            tempPose = self.myModel.predict(inputImage)
            tempGoal = goal_pose(tempPose[0], tempPose[1], tempPose[2])
            # tempGoal = goal_pose(1.4, 0.0, 0.0)
            print('{} {} {}'.format(tempPose[0], tempPose[1], tempPose[2]))
            self.client.send_goal(tempGoal)

            r.sleep()


if __name__ == '__main__':
    rospy.init_node('random_run')
    bot = RandomBot('Random')
    bot.strategy()

