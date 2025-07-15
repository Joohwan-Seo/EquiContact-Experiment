
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
import yaml

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

    def __init__(self, config,debug="False" ): 

        super().__init__('act_inference_node')
        self.config = config 
        self.debug = self.config['env']['debug']
        self.policy_params = config['policy']
        self.policy_config = {'lr': self.policy_params['lr'],
                            'num_queries': self.policy_params['chunk_size'],
                            'kl_weight': self.policy_params['kl_weight'],
                            'hidden_dim': self.policy_params['hidden_dim'],
                            'dim_feedforward': self.policy_params['dim_feedforward'],
                            'lr_backbone': self.policy_params['lr_backbone'],
                            'backbone': self.policy_params['backbone'],
                            'enc_layers': self.policy_params['enc_layers'],
                            'dec_layers': self.policy_params['dec_layers'],
                            'nheads': self.policy_params['nheads'],
                            'camera_names': self.config['env']['camera_names'],
                            }
        self.config_dict = {
            'num_epochs': self.policy_params['num_epochs'],
            'ckpt_dir': self.policy_params['ckpt_dir'],
            'episode_len': self.policy_params['episode_len'],
            'state_dim': self.policy_params['state']['dim'],
            'lr': self.policy_params['lr'],
            'policy_class': self.policy_params['class'],
            'onscreen_render': self.policy_params['onscreen_render'],
            'policy_config': self.policy_config,
            'task_name': self.policy_params['task'],
            'seed': self.policy_params['seed'],
            'temporal_agg': self.policy_params['temporal_agg'],
            'camera_names': self.config['env']['camera_names'],
            'real_robot': not self.policy_params['is_sim']
        }
        self.msg_types = {"Image": Image, "Twist": Twist, "DiffAction": DiffAction, "CompACT": CompAct, "RoboData": RoboData}
        self.pubs = {}
        self.init_sensors()
        self.init_policy()
        dt = 1/self.config['env']['frequency']
        max_timesteps = self.config['env']['max_episode_steps']
        temporal_agg = self.policy_params['temporal_agg']
        state_dim = self.policy_params['state']['dim']
        if temporal_agg:
                # Initialize all_time_actions if it's the first call
            self.all_time_actions = torch.zeros([max_timesteps, max_timesteps + self.policy_params['chunk_size'], state_dim]).cuda()
        self.exec_loop = self.create_timer(dt, self.inference, callback_group=ReentrantCallbackGroup())
        self.t = 0
        self.obs = {}
        
        self.counter = 0

    def init_sensors(self): 
        inputs = self.config['env']['inputs']
        if "camera_names" in self.config['env'].keys(): 
            self.bridge = cvb.CvBridge()
            self.imgs = {}

        for k,v in inputs.items():
            if v['type'] == "rgb":
                self.imgs[k] = np.zeros(v['size'], dtype=np.float32)
                print(f"Camera: {k}, topic: {v['topic']}, type: {v['msg_type']}")
                self.create_subscription(self.msg_types[v['msg_type']], v['topic'], 
                                         partial(self.img_cb,camera_id=k, size=v['size']), 10)
            elif v['type'] == "proprio":
                self.state = np.zeros(v['size'], dtype=np.float32)
                self.create_subscription(self.msg_types[v['msg_type']], v['topic'], 
                                         partial(self.state_cb,key=k, type=v['msg_type']), 10)
        outputs = self.config['env']['outputs']
        for k,v in outputs.items():
            print(v['msg_type'])
            pub_i = self.create_publisher(self.msg_types[v['msg_type']], v['topic'], 10)
            self.pubs[k] = pub_i
            
    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy

    def init_policy(self):
        ckpt_path = os.path.join(self.policy_params['ckpt_dir'], self.policy_params['ckpt_name']) 
        self.policy = self.make_policy(self.policy_params['class'], self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(self.policy_params['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

    def img_cb(self, msg,camera_id, size):
        curr_images = []
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for cam_name in self.camera_names:
              # Using `image`, not `obs['images']`
        curr_image = cv2.resize(img, (640, 480))
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)

        # curr_image = np.array(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        
        self.obs[camera_id] = curr_image
    
    def state_cb(self, msg, key, type): 
        data = self.msg2arr(msg, type)
        format = self.config["env"]["inputs"][key]['format']
        self.state_sep = { }
        for k, v in format.items():
            self.state_sep[k] = data[v['ind'][0]:v['ind'][1]+1]
            if v["locality"] == "init_pose":
                if self.counter == 0: 
                    self.init_pose = data[v['ind'][0]:v['ind'][1]+1]
                    self.counter += 1
                self.state_sep[k] = self.preprocess_state(data[v['ind'][0]:v['ind'][1]+1], v['type'],v['locality'])

        self.obs['state'] = np.array(list(self.state_sep.values())).reshape((-1,)).astype(np.float32)


    def arr2msg(self, arr, type):
        if type == "DiffAction":
            msg = DiffAction()
            twists = msg.actions
            act = np.tile(arr, (5, 1))
            for i in range(act.shape[0]):

                twist_i = Twist()
                twist_i.linear.x = float(act[i, 0])
                twist_i.linear.y = float(act[i, 1])
                twist_i.linear.z = float(act[i, 2])

                twist_i.angular.x = float(act[i, 3])
                twist_i.angular.y = float(act[i, 4])
                twist_i.angular.z = float(act[i, 5])
                twists.append(twist_i)
            return msg
        elif type == "CompACT": 
            msg = CompAct()
            pose = arr[:6]
            gains = arr[6:]
            gains = np.power(10, gains)
            print(f"gains: {gains}")
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
            return msg    
    
    def msg2arr(self, msg, type):
        if type == "RoboData": 
            data = np.zeros((12,), dtype=np.float32)
            data[0] = msg.ppos.linear.x
            data[1] = msg.ppos.linear.y
            data[2] = msg.ppos.linear.z
            data[3] = msg.ppos.angular.x
            data[4] = msg.ppos.angular.y
            data[5] = msg.ppos.angular.z
            data[6] = msg.wrench.linear.x
            data[7] = msg.wrench.linear.y
            data[8] = msg.wrench.linear.z
            data[9] = msg.wrench.angular.x
            data[10] = msg.wrench.angular.y
            data[11] = msg.wrench.angular.z
            return data 
        elif type == "Twist":
            data = np.zeros((6,), dtype=np.float32)
            data[0] = msg.linear.x
            data[1] = msg.linear.y
            data[2] = msg.linear.z
            data[3] = msg.angular.x
            data[4] = msg.angular.y
            data[5] = msg.angular.z
            return data
    
    def postprocess_actions(self, actions, current_pose, type="geometric",locality="global",debug=False): 
        if type == "geometric": 
            g_delta = se3_expmap(actions)
            g_current = cart2se3(current_pose)
            g_command = g_current @ g_delta
            cart_command = se32cart(g_command)
            
        elif type == "cartesian": 
            cart_command = actions
            if locality == "init_pose": 
                cart_command_p = R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix() @ cart_command[:3] + self.init_pose[:3]
                cart_command_a_rot = R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix() @ R.from_euler('xyz', cart_command[3:6], degrees=True).as_matrix()
                cart_command_a_euler = R.from_matrix(cart_command_a_rot).as_euler('xyz', degrees=True)
                cart_command = np.concatenate((cart_command_p, cart_command_a_euler, cart_command[6:]), axis=0)
            elif locality == "current_pose": 
                cart_command_p = R.from_euler('xyz', current_pose[3:6], degrees=True).as_matrix() @ cart_command[:3] + current_pose[:3]
                cart_command_a_rot = R.from_euler('xyz', current_pose[3:6], degrees=True).as_matrix() @ R.from_euler('xyz', cart_command[3:6], degrees=True).as_matrix()
                cart_command_a_euler = R.from_matrix(cart_command_a_rot).as_euler('xyz', degrees=True)
                cart_command = np.concatenate((cart_command_p, cart_command_a_euler, cart_command[6:]), axis=0)

        if debug: 
            print(f"command: {cart_command}")
        else:
            print(f"Non debug command: {cart_command}")
            self.pubs['action'].publish(self.arr2msg(cart_command, type="CompACT"))
            # self.act_pub.publish(self.arr2msg(cart_command, type="compact"))

    def preprocess_state(self, obs_pose, type="geometric", locality="global"): 
        if type == "geometric":
            init_g = cart2se3(self.init_pose)
            obs_g = cart2se3(obs_pose)
            twist = get_rel_twist(init_g, obs_g)
            return twist
        elif type == "cartesian": 
            ang = obs_pose[3:6]
            rot = R.from_euler( 'xyz',ang, degrees=True)
            rotmat = rot.as_matrix()
            if locality == "init_pose" : 
                obs_pose_p = np.linalg.inv(R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix()) @ (obs_pose[:3] - self.init_pose[:3])
                obs_pose_a_rot = np.linalg.inv(R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix()) @ R.from_euler('xyz', obs_pose[3:6], degrees=True).as_matrix()
                obs_pose_a_euler = R.from_matrix(obs_pose_a_rot).as_euler('xyz', degrees=True)
                obs_pose = np.concatenate((obs_pose_p, obs_pose_a_euler), axis=0)
            return obs_pose
        else: 
            return obs_pose

    def format_obs(self): 
        obs = collections.OrderedDict()
        obs['state'] = self.obs['state']
        # Stack images from wrist and wrist_back
        obs['images'] = torch.cat([self.obs[camera_name]for camera_name in self.config['env']['camera_names']], dim=0).unsqueeze(0)  
        return obs 
    
    def inference(self):
        set_seed(1000)
        num_queries = self.policy_params['chunk_size']
        if self.policy_params['temporal_agg']: 
            query_freq = 1
        else:
            query_freq = self.policy_params['chunk_size']
        t = self.t
        pre_process = lambda s_ppos: (s_ppos - self.stats['ppos_mean']) / self.stats['ppos_std'] 
        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        
        with torch.inference_mode():
            obs = self.format_obs()
            state = obs['state']
            print(f"Current state: {state}")
            curr_image = obs['images']
            state_copy = copy.deepcopy(state)
            state_in = pre_process(state_copy)
            state_in = torch.from_numpy(state_in).float().cuda().unsqueeze(0)
            if self.config['policy']['class'] == "ACT":
                if t % query_freq == 0:
                    all_actions = self.policy(state_in, curr_image)

                if self.policy_params['temporal_agg']:
                    self.all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = self.all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else: 
                    raw_action = all_actions[:, t % query_freq]
            elif self.config['policy_class'] == "CNNMLP":
                raw_action = self.policy(state, curr_image)
            else:
                raise NotImplementedError
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        print(f"Raw action: {action}")
        action_format = self.config['env']['outputs']['action']['format']

        self.postprocess_actions(action, self.obs['state'], 
                                 type =action_format['pose_target']['type'],
                                 locality = action_format['pose_target']['locality'],
                                 debug = self.debug)
        
def main(args): 
    debug = args.debug 
    config_path = args.config 
    rclpy.init(args=None)
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    act_inference = ACTInference(config, debug=debug)
    rclpy.spin(act_inference)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/inference_config.yaml', help='Path to the config file')
    parser.add_argument('--debug', default=True, help='Debug mode')
    args = parser.parse_args()
    main(args)


