import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from camera_interfaces.msg import DiffAction

### Camera Interface 
import cv2  
import cvbridge as cvb 
import numpy as np


# class RealEnv(Node): 

#     def __init__(self, robot_ip, )