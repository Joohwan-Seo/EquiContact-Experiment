#ros and robot imports
import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from camera_interfaces.msg import DiffAction,CompAct, RoboData
import cv_bridge as cvb
import cv2
import argparse
from functools import partial
from sensor_msgs.msg import Image
from geometric_func import *
from collections import deque
import time

from neuromeka import IndyDCP3

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import copy

# from rclpy.qos import QoSProfile
from act.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from act.utils import load_data # data functions
from act.utils import sample_box_pose, sample_insertion_pose # robot functions
from act.utils import compute_dict_mean, set_seed, detach_dict # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy
from act.visualize_episodes import save_videos


from act.sim_env import BOX_POSE
import IPython
import collections
import json
import cv2

class ACTInference(Node): 

    def __init__(self, debug=False):
        super().__init__('act_inference_hardcoded_node')
        self.debug = debug
        self.indy = IndyDCP3("10.208.112.143")
        self.ckpt_dir = '/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/act/ckpt_dir/real_peg_2view'
        self.task_name = "real_peg_2view"
        
        self.imgs = {
            "wrist": None,
            "wrist_back": None
        }
        self.policy_config = { 
            'lr': 1e-4,
            'num_queries':100,
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'state_dim': 6,
            'action_dim': 6,
            'nheads': 8,
            'camera_names': ['wrist', 'wrist_back'],
        }
        self.config = {
            'num_epochs': 8000,
            'ckpt_dir': self.ckpt_dir,
            'episode_len': 5000,
            'state_dim': 6,
            'lr': 1e-4,
            'policy_class': 'ACT',
            'onscreen_render': False,
            'policy_config': self.policy_config,
            'task_name': self.task_name,
            'seed': 0,
            'temporal_agg': True,
            'camera_names': ["wrist", "wrist_back"],
            'real_robot': True
        }
        self.frequency = 20
        self.dt = 1/self.frequency 
        self.obs  = { 
            'wrist': None,
            'wrist_back': None, 
            'pose': None,
            'ft': None,
        }

        #### Initializing Subscribers, Publishers & Timers ####### 
        self.img_bridge = cvb.CvBridge()
        self.camera_subs = {
            "wrist": self.create_subscription(Image, "/realsense/camera/color/image_raw", partial(self.get_imgs, camera_id="wrist"), 10), 
            "wrist_back": self.create_subscription(Image, "/arducam/color/image_raw",partial(self.get_imgs, camera_id="wrist_back"), 10)
        }
        self.init_policy()
        # self.pose_timer = self.create_timer(1/50, self.get_pose, callback_group=ReentrantCallbackGroup())
        # time.sleep(3)
        self.exec = self.create_timer(self.dt, self.eval, callback_group=ReentrantCallbackGroup())
       
        self.act_pub = self.create_publisher(CompAct, "/indy/act2", 10)
        self.t = 0 
        self.start = False 
        self.all_time_actions = []

    def init_policy(self): 
        ckpt_name = "policy_best.ckpt"
        ckpt_path = os.path.join(self.config['ckpt_dir'], ckpt_name)
        self.policy = ACTPolicy(self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print("Loading status: ", loading_status)
        self.policy.cuda()
        self.policy.eval()
        stats_path = os.path.join(self.config['ckpt_dir'], "dataset_stats.pkl")
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        print(f"State mean: {self.stats['ppos_mean']}, \nAction mean: {self.stats['action_mean']}")
        print(f"State std: {self.stats['ppos_std']}, \nAction std: {self.stats['action_std']}")

    def get_imgs(self, msg, camera_id):
        img = self.img_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        curr_image = cv2.resize(img, (640, 480))
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_image_tensor = torch.from_numpy(curr_image / 255.0).float().cuda()
        self.obs[camera_id] = curr_image_tensor

    def get_pose(self): 
        pose = np.array(self.indy.get_control_data()['p'])
        ft_dict = self.indy.get_ft_sensor_data()
        ft = np.array([ft_dict['ft_Fx'], ft_dict['ft_Fy'], ft_dict['ft_Fz'], ft_dict['ft_Tx'], ft_dict['ft_Ty'], ft_dict['ft_Tz']])
        self.obs['pose'] = pose
        self.obs['ft'] = ft
        
    def get_obs(self): 
        pose = np.array(self.indy.get_control_data()['p'])
        ft_dict = self.indy.get_ft_sensor_data()
        ft = np.array([ft_dict['ft_Fx'], ft_dict['ft_Fy'], ft_dict['ft_Fz'], ft_dict['ft_Tx'], ft_dict['ft_Ty'], ft_dict['ft_Tz']])
        # pose = self.obs['pose']
        # ft = self.obs['ft']
        if pose is None or ft is None:
            print("Pose or FT data not available yet.")
            return None
        state = np.concatenate((pose, ft))
        imgs = torch.stack([self.obs["wrist"], self.obs["wrist_back"]], dim=0).unsqueeze(0)
        print(f"Image_shapes: {imgs.shape},\n State shape: {state.shape}")
        obs = collections.OrderedDict()
        obs['state'] = pose
        obs['pvel'] = None 
        obs['effort'] = None
        obs['images'] = imgs
        return obs 
        
    def arr2msg(self, act): 
        msg = CompAct()
        pose = act[:6]
        # gains = act[6:]
        # gains = np.power(10, gains)
        # print(f"gains: {gains}")
        msg.cart_pose.linear.x = float(pose[0])
        msg.cart_pose.linear.y = float(pose[1])
        msg.cart_pose.linear.z = float(pose[2])
        msg.cart_pose.angular.x = float(pose[3])
        msg.cart_pose.angular.y = float(pose[4])
        msg.cart_pose.angular.z = float(pose[5])
        # msg.ad_gains.linear.x = float(gains[0])
        # msg.ad_gains.linear.y = float(gains[1])
        # msg.ad_gains.linear.z = float(gains[2])
        # msg.ad_gains.angular.x = float(gains[3])
        # msg.ad_gains.angular.y = float(gains[4])
        # msg.ad_gains.angular.z = float(gains[5])
        msg.ad_gains.linear.x = float(1000)
        msg.ad_gains.linear.y = float(1000)
        msg.ad_gains.linear.z = float(300)
        msg.ad_gains.angular.x = float(1000)
        msg.ad_gains.angular.y = float(1000)
        msg.ad_gains.angular.z = float(1000)
        return msg
    
    def eval(self): 
        if self.t >= self.config['episode_len']:
            print("Reached maximum timesteps")
            return
        set_seed(0)
        
        #### Fetching some useful parameters ####
        state_dim = self.config['state_dim']
        policy_config = self.config['policy_config']
        camera_names = self.config['camera_names']
        max_timesteps = self.config['episode_len']
        task_name = self.config['task_name']
        temporal_agg = self.config['temporal_agg']
        real_robot = self.config['real_robot']
        policy_class = self.config['policy_class']
        onscreen_render = self.config['onscreen_render']
        query_freq = policy_config['num_queries']
        if temporal_agg:
            query_freq = 1
            num_queries = policy_config['num_queries']
        self.all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()
        #### Preprocessing Transforms ####
        pre_process = lambda s_ppos: (s_ppos - self.stats['ppos_mean']) / self.stats['ppos_std'] 
        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        #### Inference Loop ####
        with torch.inference_mode(): 
            obs = self.get_obs()
            state = obs['state']
            processed_state = pre_process(state)
            processed_state = torch.from_numpy(processed_state).float().cuda().unsqueeze(0)
            img = obs['images']
            if self.t % query_freq == 0:
                all_actions = self.policy(processed_state, img)

            if temporal_agg:
                self.all_time_actions[[self.t], self.t:self.t+num_queries] = all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else: 
                raw_action = all_actions[:, self.t % query_freq]
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        if self.debug: 
            print(f"Debug, Raw action: {raw_action}")
            print(f"Debug, Post processed action: {action}")
        else: 
            print(f"Action: {action}")
            action[3] = 180 
            self.act_pub.publish(self.arr2msg(action))
        self.t += 1
def main(): 
    rclpy.init()
    act_inference = ACTInference(debug=False)
    rclpy.spin(act_inference)

if __name__ == "__main__": 
    main()