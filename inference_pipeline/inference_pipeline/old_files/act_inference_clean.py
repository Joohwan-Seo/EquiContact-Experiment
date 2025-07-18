#ros and robot imports
import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from camera_interfaces.msg import DiffAction
import cv_bridge as cvb
import cv2
import argparse
from functools import partial
from sensor_msgs.msg import Image
from geometric_func import cart2se3, get_rel_twist, se3_logmap, so3_logmap
from collections import deque
import time

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange


# from rclpy.qos import QoSProfile
from act.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from act.utils import load_data # data functions
from act.utils import sample_box_pose, sample_insertion_pose # robot functions
from act.utils import compute_dict_mean, set_seed, detach_dict # helper functions
from act.policy import ACTPolicy, CNNMLPPolicy
from act.visualize_episodes import save_videos
from act.act_inference_test import inference

from act.sim_env import BOX_POSE
import IPython
import collections
import json
import cv2


class ACTInference(Node):
    def __init__(self, 
                 camera_topics={"wrist":"/realsense/camera/color/image_raw", "wrist_back":"/arducam/color/image_raw"}, 
                 indy_ip ="10.208.112.115",
                 ckpt_dir = '/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/act/ckpt_dir/real_peg_local2',
                 ckpt_name = 'policy_best.ckpt',
                 policy_class = 'ACT',
                 onscreen_render = False,
                 task_name = 'real_peg_local2',
                 lr = 1e-4,
                 chunk_size = 100,
                 kl_weight = 10,
                 hidden_dim = 512,
                 dim_feedforward = 3200,
                 lr_backbone = 1e-5,
                 backbone = 'resnet18',
                 camera_names = ['wrist', 'wrist_back'],
                 num_epochs = 8000,
                 episode_len = 1000,
                 seed = 0,
                 temporal_agg = True,
                 is_sim = False,
                 ):
        super().__init__('act_inference')

        self.camera_topics = camera_topics 
        self.indy_ip = indy_ip

        set_seed(1)
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.policy_class = policy_class
        self.onscreen_render = onscreen_render
        self.task_name = task_name
        self.camera_names = camera_names

        state_dim = 12

        if self.policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            self.policy_config = {'lr': lr,
                                'num_queries':chunk_size,
                                'kl_weight': kl_weight,
                                'hidden_dim': hidden_dim,
                                'dim_feedforward': dim_feedforward,
                                'lr_backbone': lr_backbone,
                                'backbone': backbone,
                                'enc_layers': enc_layers,
                                'dec_layers': dec_layers,
                                'nheads': nheads,
                                'camera_names': camera_names,
                                }
        elif self.policy_class == 'CNNMLP':
            self.policy_config = {'lr': lr, 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                                'camera_names': camera_names,}
        else:
            raise NotImplementedError

        self.num_epochs = num_epochs

        self.config = {
            'num_epochs': num_epochs,
            'ckpt_dir': ckpt_dir,
            'episode_len': episode_len,
            'state_dim': state_dim,
            'lr': lr,
            'policy_class': policy_class,
            'onscreen_render': onscreen_render,
            'policy_config': self.policy_config,
            'task_name': task_name,
            'seed': seed,
            'temporal_agg': temporal_agg,
            'camera_names': camera_names,
            'real_robot': not is_sim
        }

        self.frequency = 50
        self.dt = 1/self.frequency
        self.obs_pose = None
        self.obs_img = {}
        self.initialize_sensors()
        self.init_model()
        self.inference_loop = self.create_timer(self.dt, self.eval_bc, callback_group=ReentrantCallbackGroup())
        self.t = 0
        self.all_time_actions = None

        self.counter = 0
        ### move to home position
        # self.act_pub.publish(self.arr2msg(np.array([350, -186.5, 522.1, 180, 0, 90])))

    def initialize_sensors(self): 
        self.img_subs = {camera_id: self.create_subscription(Image, topic, partial(self.img_cb, camera_id=camera_id), 10, callback_group=ReentrantCallbackGroup()) for camera_id, topic in self.camera_topics.items()}
        self.pose_sub = self.create_subscription(Twist, "/indy/pose", self.get_pose, 10, callback_group=ReentrantCallbackGroup())
        self.act_pub = self.create_publisher(DiffAction, "/indy/act", 10, callback_group=ReentrantCallbackGroup())

        self.bridge = cvb.CvBridge()

    def get_pose(self, msg): 
        pose_msg = msg 
        self.obs_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
        
        if self.counter == 0:
            self.init_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
            self.obs_pose = np.zeros((6,))
            self.counter += 1
        else: 
            self.obs_pose_se3 = cart2se3(self.obs_pose)
            self.init_pose_se3 = cart2se3(self.init_pose)
            twist = get_rel_twist(self.obs_pose_se3, self.init_pose_se3)
            self.obs_pose = twist
            
    def img_cb(self, msg, camera_id):
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
        self.obs_img[camera_id]= curr_image
        
    def arr2msg(self, act):
        msg = DiffAction()
        twists = msg.actions
        act = np.tile(act, (5, 1))
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

    def sensor_loops_test(self): 
        if len(self.pose_queue) != 0 and len(self.img_queues["realsense"]) != 0: 
            
            print(len(self.pose_queue), self.pose_queue[0])
            print(len(self.img_queues["realsense"])) 
            plt.imshow(self.img_queues["realsense"][0])
            plt.show()

    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy

    def get_obs(self):
        obs = collections.OrderedDict()
        obs['ppos'] = self.obs_pose
        obs['pvel'] = None
        obs['effort'] = None
        # Stack images from wrist and wrist_back
        obs['images'] = torch.cat([self.obs_img[camera_name] for camera_name in self.camera_names], dim=0).unsqueeze(0)
        return obs
    
    def init_model(self):
        
        ckpt_name = self.ckpt_name
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        self.policy = self.make_policy(self.policy_class, self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(self.ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # self.stats['ppos_mean'] = self.stats['ppos_mean'] + np.array([25.4, 8*25.4, 0, 0, 0, 0])
        # self.stats['action_mean'] =self.stats['action_mean'] +  np.array([25.4, 8*25.4, 0, 0, 0, 0])

    def eval_bc(self):
        # check if max timesteps reached
        if self.t >= self.config['episode_len']:
            print("Reached maximum timesteps")
            return

        print("Evaluating BC")
        set_seed(1000)
        ckpt_dir = self.config['ckpt_dir']
        state_dim = self.config['state_dim']
        real_robot = self.config['real_robot']
        policy_class = self.config['policy_class']
        onscreen_render = self.config['onscreen_render']
        policy_config = self.config['policy_config']
        camera_names = self.config['camera_names']
        max_timesteps = self.config['episode_len']
        task_name = self.config['task_name']
        temporal_agg = self.config['temporal_agg']
        onscreen_cam = 'angle'

        ### load policy and stats
        pre_process = lambda s_ppos: (s_ppos - self.stats['ppos_mean']) / self.stats['ppos_std'] 
        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        
        query_frequency = policy_config['num_queries']
        if temporal_agg:
            query_frequency = 1
            num_queries = policy_config['num_queries']

        max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

        ### evaluation loop
        if temporal_agg:
            if self.all_time_actions is None:
                # Initialize all_time_actions if it's the first call
                self.all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        ppos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        init_time = time.time()

        with torch.inference_mode():
            t = self.t
            print(f"t: {t}")

            ### process previous timestep to get ppos and image_list
            obs = self.get_obs()
            ppos_numpy = np.array(obs['ppos'])
            print("ppos_numpy:", ppos_numpy)
            ppos = pre_process(ppos_numpy)

            ppos = torch.from_numpy(ppos).float().cuda().unsqueeze(0)
            ppos_history[:, t] = ppos
            curr_image = obs['images']

            ### query policy
            if self.config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = self.policy(ppos, curr_image)
                if temporal_agg:
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
                    raw_action = all_actions[:, t % query_frequency]
            elif self.config['policy_class'] == "CNNMLP":
                raw_action = self.policy(ppos, curr_image)
            else:
                raise NotImplementedError

            ### post-process and publish
            self.publish(raw_action, post_process)

            self.t += 1
    
    def publish(self, raw_action, post_process):
        # post_process
        raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)
        target_ppos = action
        pred_time = time.time()

        # publish action
        self.act_pub.publish(self.arr2msg(target_ppos))

def main(): 
    rclpy.init()
    act_inference = ACTInferece()
    rclpy.spin(act_inference)

if __name__ == "__main__": 
    main()
