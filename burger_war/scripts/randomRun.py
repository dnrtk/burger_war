#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This is rumdom run node.
subscribe No topcs.
Publish 'cmd_vel' topic. 
mainly use for simple sample program

by Takuya Yamaguhi.
'''

import random
import json
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import JointState
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

import tf
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3

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
# 3Point=0.60
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
# 5Point=1.00



def euler_to_quaternion(euler):
    q = tf.transformations.quaternion_from_euler(euler.x, euler.y, euler.z)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

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
        self.mapList['FieldMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['MyQrMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['EnemyQrMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['NoneQrMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['EnemyMap'] =  np.zeros(FIELD_SIZE)
        self.mapList['MyPositionMap'] =  np.zeros(FIELD_SIZE)

        self.map_data = np.zeros(FIELD_SIZE)
        self.myPositionMap = np.zeros(FIELD_SIZE)
        # velocity publisher
        self.vel_pub = rospy.Publisher('cmd_vel', Twist,queue_size=1)
        self.map_sub = rospy.Subscriber('map', OccupancyGrid, self.mapCallback)
        # self.gCostMap_sub = rospy.Subscriber('move_base/global_costmap/costmap', OccupancyGrid, self.globalCostmapCallback)
        self.warState_sub = rospy.Subscriber('war_state', String, self.warStateCallback)
        self.odom = Odometry()
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odometryCallback)
    
    def mapCallback(self, data):
        # print('MapCB')
        self.map = data
        self.map_data = occupancyGrid_to_ndarray(data)
        # np.savetxt('./map.csv', self.map_data)

    def globalCostmapCallback(self, data):
        # print('GlobalCostmapCB')
        self.gCostMap = data
        self.gCostMap_data = occupancyGrid_to_ndarray(data)

    def warStateCallback(self, data):
        # print('WarStateCB')
        self.warState = data.data
        type(self.warState)
        d = json.loads(self.warState)
        # print(d['players'])
        # print(d['scores'])
        # for target in d['targets']:
        #     print(target)
        # print(int(d['targets'][0]['point']))
        
    def odometryCallback(self, data):
        # print('OdometryCB')
        self.odom = data

        self.poseCovMatrix = np.array(self.odom.pose.covariance).reshape((6, 6))
        self.twistCovMatrix = np.array(self.odom.twist.covariance).reshape((6, 6))
        # print(self.poseCovMatrix)
        # print(self.twistCovMatrix)

        # print('x : {}'.format(self.odom.pose.pose.position.x))
        # print('y : {}'.format(self.odom.pose.pose.position.y))
        tempX = int((self.odom.pose.pose.position.x + 2.0) / 4.0 * FIELD_SIZE[1])
        tempY = int(80 - (self.odom.pose.pose.position.y + 2.0) / 4.0 * FIELD_SIZE[0])
        # print('{} {}'.format(tempX, tempY))
        tempVector3 = quaternion_to_euler(self.odom.pose.pose.orientation)
        tempRotate = tempVector3.z / math.pi * 180
        # print(tempRotate)
        
        # self.myPositionMap[tempY][tempX] = 1.0
        # self.myPositionMap[tempY][tempX] = 0.5
        trans = cv2.getRotationMatrix2D((myRobotMapCenterX, myRobotMapCenterY), tempRotate , 1.0)
        tempMyRobotMap = cv2.warpAffine(myRobotMap, trans, myRobotMap.shape)

        # 自ロボ位置と射程を描画
        self.myPositionMap = np.zeros(FIELD_SIZE)
        self.myPositionMap += self.map_data
        offsetX = tempX - myRobotMapCenterX
        offsetY = tempY - myRobotMapCenterY
        for indexY, valueY in enumerate(tempMyRobotMap):
            for indexX, valueX in enumerate(valueY):
                self.myPositionMap[indexY + offsetY][indexX + offsetX] += valueX


    def calcTwist(self):
        value = random.randint(1,1000)
        # value = 1001
        if value < 250:
            x = 0.2
            th = 0
        elif value < 500:
            x = -0.2
            th = 0
        elif value < 750:
            x = 0
            th = 1
        elif value < 1000:
            x = 0
            th = -1
        else:
            x = 0
            th = 0
        twist = Twist()
        twist.linear.x = x; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = th
        return twist

    def strategy(self):
        r = rospy.Rate(1) # change speed 1fps

        plt.figure()

        target_speed = 0
        target_turn = 0
        control_speed = 0
        control_turn = 0

        while not rospy.is_shutdown():
            plt.imshow(self.myPositionMap,interpolation='nearest',vmin=0,vmax=1,cmap='jet')
            plt.pause(0.001)

            twist = self.calcTwist()
            # print(twist)
            self.vel_pub.publish(twist)

            r.sleep()


if __name__ == '__main__':
    rospy.init_node('random_run')
    bot = RandomBot('Random')
    bot.strategy()

