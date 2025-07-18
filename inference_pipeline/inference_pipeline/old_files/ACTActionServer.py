import rclpy 
from rclpy.node import Node
from rclpy.action import ActionServer

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from camera_interfaces.msg import DiffAction, CompAct, RoboData, ActActivity
from camera_interfaces.action import ACTAction
import cv_bridge as cvb
import cv2
import argparse
from functools import partial
from sensor_msgs.msg import Image
from geometric_func import *
from collections import deque
import time
import yaml

import torch
import gc
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


class ACTActionServer(Node):

    def __init__(self, config): 
        super().__init__("act_action_server")
        self._action_server = ActionServer(
            self,
            'act_inference',
            ACTAction,
            self.action_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        
        ##### Initialize Policy Parameters
        self.config = config 
        self.debug = False
        
        self.current_task = 'pick' # initialization

        self.task_config = {}
        self.eval_config = {}
        self.train_config = {}

        self.policy = None

        for task in ['pick', 'place']:           
            self.task_config[task] = config[task]['task_config']
            self.eval_config[task] = config[task]['eval_config']
            self.train_config[task] = config[task]['train_config']

        ##### NOTE(JS): Later to be changed using the response from service ######
        self.reference_frame = np.array(self.train_config[task]['reference_frame'])
        ##########################################################################
       
        self.msg_types = {"Image": Image, "Twist": Twist, "DiffAction": DiffAction, "CompACT": CompAct, "RoboData": RoboData}
        self.pubs = {}

        self.dt = 1/self.eval_config[task]['frequency'] # 1/30
        
        self.t = 0
        self.obs = {}

        # initialization parameters for the cameras; assuming that pick and place have the same camera names
        self.camera_names = self.task_config['pick']['camera_names']
        self.crop_image = False
        self.mask_image = False
        
        self.init_sensors()

        self.active = False
        
        self.counter = 0

        self.get_logger().info("ACT Inference Server is ready to receive requests.")
    
    def init_sensors(self): #NOTE(JS): Done
        # initialize camera
        if self.camera_names is not None:
            self.imgs = {}
            self.bridge = cvb.CvBridge()

        for cam_name in self.camera_names:
            self.imgs[cam_name] = np.zeros((3, 480, 640), dtype=np.float32)

            if cam_name == "realsense":
                topic = "/realsense/camera/color/image_raw"
            elif cam_name == "arducam":
                topic = "/arducam/color/image_raw"

            self.create_subscription(Image, topic, partial(self.img_cb, camera_id=cam_name), 10, callback_group=ReentrantCallbackGroup())

        # intiailize proprioceptive sensors
        self.create_subscription(RoboData, "/indy/state", self.state_cb, 10, callback_group=ReentrantCallbackGroup())
        ## initialize binary subscriber 
        self.create_subscription(ActActivity, "act_activity", self.activity_cb, 10, callback_group=ReentrantCallbackGroup())
        # initialize publishers
        self.pubs['action'] = self.create_publisher(CompAct, "/indy/act2", 10)
    
    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy

    def init_policy(self, task):
        ckpt_path = os.path.join(self.eval_config[task]['ckpt_dir'], self.eval_config[task]['ckpt_name'])         
        self.policy = self.make_policy("ACT", self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(self.eval_config[task]['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.pre_process = lambda s_ppos: (s_ppos - self.stats['state_mean']) / self.stats['state_std'] 
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

    def img_cb(self, msg, camera_id): 
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        curr_image = cv2.resize(img, (640, 480))
        
        if self.crop_image:
            curr_image = self.center_crop(curr_image, ratio=self.crop_ratio)
        
        if self.mask_image:
            curr_image = self.boundary_mask(curr_image, mask_width_frac=self.mask_ratio)

        curr_image = rearrange(curr_image, 'h w c -> c h w')
        
        self.obs[camera_id] = curr_image
    
    def state_cb(self, msg): 
        self.obs["ppos"] = np.array([msg.ppos.linear.x, msg.ppos.linear.y, msg.ppos.linear.z,
                                     msg.ppos.angular.x, msg.ppos.angular.y, msg.ppos.angular.z], dtype=np.float32)
        self.obs["pvel"] = np.array([msg.pvel.linear.x, msg.pvel.linear.y, msg.pvel.linear.z,
                                     msg.pvel.angular.x, msg.pvel.angular.y, msg.pvel.angular.z], dtype=np.float32)
        self.obs["FT"] = np.array([msg.ft.linear.x, msg.ft.linear.y, msg.ft.linear.z,
                                       msg.ft.angular.x, msg.ft.angular.y, msg.ft.angular.z], dtype=np.float32)
        
    def activity_cb(self, msg): 
        self.active = msg.active
        self.reference_frame = np.array(msg.reference_frame)
        task = msg.task

        if task == self.current_task:
            return
        
        self.unload_model()

        self.load_model(task)

        self.current_task = task

        print(self.active)

    def load_model(self, task):
        self.state_dim = self.task_config[task]['state_dim']
        self.action_dim = self.task_config[task]['action_dim']
        self.num_queries = self.task_config[task]['num_queries']
        self.state_category = self.task_config[task]['state_category']
        self.action_category = self.task_config[task]['action_category']
        self.policy_class = self.task_config[task]['policy_class']

        self.camera_names = self.task_config[task]['camera_names']

        self.crop_image = self.train_config[task]['crop_image']
        self.crop_ratio = self.train_config[task]['crop_ratio']
        self.mask_image = self.train_config[task]['mask_image']
        self.mask_ratio = self.train_config[task]['mask_ratio']

        self.episode_len = self.eval_config[task]['episode_len']
        self.temporal_agg = self.eval_config[task]['temporal_agg']

        # fixed policy config
        
        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        self.policy_config = {'lr': float(self.train_config[task]['lr']),
                        'num_queries': self.num_queries,
                        'kl_weight': self.train_config[task]['kl_weight'],
                        'hidden_dim': self.task_config[task]['hidden_dim'],
                        'dim_feedforward': self.task_config[task]['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': self.camera_names,
                        'state_dim': self.state_dim,
                        'action_dim': self.action_dim,
                        }

        self.init_policy(task)

        buffer = 100
        if self.temporal_agg:
            # Initialize all_time_actions if it's the first call
            self.all_time_poses_np = np.zeros((self.episode_len + buffer, self.episode_len + buffer + self.num_queries,4,4))
            self.all_time_gripper_np = np.zeros((self.episode_len + buffer, self.episode_len + buffer + self.num_queries, 1))
            self.all_time_gains_np = np.zeros((self.episode_len + buffer, self.episode_len + buffer + self.num_queries, 6))
        pass

    def unload_model(self):
        if self.policy is not None:
            del self.policy
            self.policy = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            self.get_logger().info("Unloaded previous model and cleared CUDA cache")   

    def shoot_action(self, pose, gains, gripper, debug = False):
        msg = CompAct()
        msg.cart_pose.linear.x = float(pose[0])
        msg.cart_pose.linear.y = float(pose[1])
        msg.cart_pose.linear.z = float(pose[2])
        msg.cart_pose.angular.x = float(pose[3])
        msg.cart_pose.angular.y = float(pose[4])
        msg.cart_pose.angular.z = float(pose[5])
        msg.ad_gains.linear.x = float(gains[0])
        msg.ad_gains.linear.y = float(gains[1])
        msg.ad_gains.linear.z = float(gains[2])
        msg.ad_gains.angular.x = float(gains[3])
        msg.ad_gains.angular.y = float(gains[4])
        msg.ad_gains.angular.z = float(gains[5])

        if gripper >= 0.3:
            gripper_value = int(1)
        else:
            gripper_value = int(0)

        msg.gripper_state = gripper_value
        if debug:
            print(f"Action: {pose}, {gains}, {gripper}")
        else:
            print(f"Non-debug Action: {pose}, {gains}, {gripper}")
            # print(f"Non-debug Action: {gripper}")
            self.pubs['action'].publish(msg)
    
    def process_actions(self, actions):
        all_action_poses_np = np.zeros((self.num_queries, 4, 4))
        all_action_gains_np = np.ones((self.num_queries, 6)) * 1000 # default gains

        
        for action_type in self.action_category:
            if action_type == "world_pose":
                all_action_poses_np = pose_data_to_hom_matrix(actions[:, 0:6])
            
            elif action_type == "relative_pose":
                all_action_poses_np = np.expand_dims(self.current_pose, axis = 0) @ batch_pose_to_hom_matrices(actions[:, 0:6])

            elif action_type == "gains":
                all_action_gains_np = np.power(10, actions[:, 6:])
            
            elif action_type == "gripper":
                all_gripper_actions_np = actions[:, -1].reshape((-1,1))

        if self.task == 'place':
            all_gripper_actions_np = np.ones((self.num_queries, 1)) # Gripper Close
        elif self.task == 'pick':
            all_gripper_actions_np = np.ones((self.num_queries, 1))*0.01 # Gripper Open

        return all_action_poses_np, all_action_gains_np, all_gripper_actions_np

    def format_obs(self): 
        # Process states and images
        # from numpy to torch

        obs = collections.OrderedDict()
        self.current_pose = command_to_hom_matrix(self.obs['ppos'])
        states = []
        for state_type in self.state_category:
            if state_type == "world_pose":
                state = command_to_pose_data(self.obs['ppos'])

            elif state_type == "GCEV":
                gd = command_to_hom_matrix(self.reference_frame)
                g = command_to_hom_matrix(self.obs['ppos'])

                p = g[:3,3]; p_d = gd[:3,3]
                rotm = g[:3,:3]; rotm_d = gd[:3,:3]

                ep = rotm.T @ (p - p_d)
                eR = vee_map(rotm_d.T @ rotm - rotm.T @ rotm_d)

                state = np.concatenate((ep, eR), axis=0)            

            elif state_type == "FT":
                state = self.obs['FT']

            states.append(state)

        states = np.concatenate(states, axis=0)

        normalized_states = self.pre_process(states)
        assert states.shape[0] == self.state_dim, f"State dimension mismatch: {states.shape[0]} != {self.state_dim}"

        # Process images
        images = []
        for camera_name in self.camera_names:
            images.append(self.obs[camera_name])
        
        images = np.stack(images, axis=0)

        obs['state'] = torch.from_numpy(normalized_states).float().cuda().unsqueeze(0)
        obs['images'] = torch.from_numpy(images / 255.0).float().cuda().unsqueeze(0)
        return obs 
    
    def inference(self):
        set_seed(1000)
        if self.temporal_agg: 
            query_freq = 1
        else:
            query_freq = self.num_queries

        current_time = time.time()
        t = self.t

        command_pose = self.reference_frame
        gains = np.ones((6,)) * 1000

       
            # print(time.time() - current_time, t )

        if t > self.episode_len:
            if self._timer_signal is not None and not self._timer_signal.done():
                self._timer_signal.set_result(True)

            self.final_pose = command_pose
            self.final_gains = gains


        with torch.inference_mode():
            obs = self.format_obs()

            curr_image = obs['images']
            state_in = obs['state']

            # print(f"current_state: {state_in}")

            if self.policy_class == "ACT":
                if t % query_freq == 0:
                    all_actions = self.policy(state_in, curr_image).squeeze(0).cpu().numpy()
                    unnormalized_actions = self.post_process(all_actions)
                    all_action_poses_np, all_action_gains_np, all_gripper_actions_np = self.process_actions(unnormalized_actions) # updating all_time_poses_np and all_time_gains_np
                    # print(t, all_actions[0])

                if self.temporal_agg:
                    self.all_time_poses_np[[t], t:t + self.num_queries, :, :] = all_action_poses_np
                    poses_for_curr_step = self.all_time_poses_np[:, t]
                    poses_populated = (np.linalg.det(poses_for_curr_step[:,:3,:3]) != 0)
                                # print(actions_for_curr_step.shape)
                    poses_for_curr_step = poses_for_curr_step[poses_populated]

                    self.all_time_gains_np[[t], t:t + self.num_queries, :] = all_action_gains_np
                    gains_for_curr_step = self.all_time_gains_np[:, t]
                    gains_populated = np.all(gains_for_curr_step != 0, axis = 1)
                    gains_for_curr_step = gains_for_curr_step[gains_populated]

                    self.all_time_gripper_np[[t], t:t + self.num_queries, :] = all_gripper_actions_np
                    gripper_for_curr_step = self.all_time_gripper_np[:, t]
                    gripper_populated = np.all(gripper_for_curr_step != 0, axis = 1)
                    gripper_for_curr_step = gripper_for_curr_step[gripper_populated]
                    
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(poses_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()

                    #### translation : (3,) in meter, rotation: (3,3), gains: (6,)

                    translation = (poses_for_curr_step[:, :3, 3] * exp_weights.reshape((-1,1))).sum(axis=0, keepdims=False)
                    rotation_object = R.from_matrix(poses_for_curr_step[:, :3, :3])
                    rotation = rotation_object.mean(weights = exp_weights).as_matrix() # still rotation object

                    gains = (gains_for_curr_step * exp_weights.reshape((-1,1))).sum(axis=0, keepdims=False)

                    gripper = (gripper_for_curr_step * exp_weights.reshape((-1,1))).sum(axis=0, keepdims=False)
                    
                else: 
                    translation = all_action_poses_np[t % query_freq, :3, 3]
                    rotation = all_action_poses_np[t % query_freq, :3, :3]
                    gains = all_action_gains_np[t % query_freq, :]
                    gripper = all_gripper_actions_np[t % query_freq, :]

            else:
                raise NotImplementedError
            
        # convert translation, rotation and gains to the command message format
        g = np.eye(4)
        g[:3, 3] = translation
        g[:3, :3] = rotation
        command_pose = hom_matrix_to_command(g) # in mm and degrees
        self.shoot_action(command_pose, gains, gripper, debug = self.debug)

        current_time = time.time()
        self.t += 1

        return 

    def center_crop(self, image, ratio = 0.7):
        """
        center crop and resize to the original one
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * ratio), int(w * ratio)
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        cropped_image = image[start_h:start_h + new_h, start_w:start_w + new_w]
        return cv2.resize(cropped_image, (w, h))  # Resize to original size

    def boundary_mask(self, image, mask_width_frac=0.5):
        """
        Apply a soft cosine “vignette” mask around the image borders.
        mask_width_frac: fraction of width/height to taper out (0–0.5)
        """
        h, w = image.shape[:2]
        # mask widths in pixels
        mw, mh = int(w * mask_width_frac), int(h * mask_width_frac)

        # build 1D cosine taper windows
        x = np.linspace(-np.pi, np.pi, w)
        y = np.linspace(-np.pi, np.pi, h)
        wx = 0.5 * (1 + np.cos(np.clip(x / (mask_width_frac * np.pi), -np.pi, np.pi)))
        wy = 0.5 * (1 + np.cos(np.clip(y / (mask_width_frac * np.pi), -np.pi, np.pi)))

        # outer product to get 2D mask
        vignette = np.outer(wy, wx)
        vignette = vignette[:, :, np.newaxis]  # shape (h, w, 1)

        # apply mask
        masked = (image.astype(np.float32) * vignette).astype(image.dtype)
        return masked