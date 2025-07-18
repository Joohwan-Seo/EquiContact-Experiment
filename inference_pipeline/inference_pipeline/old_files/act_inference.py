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
    def __init__(self, 
                 camera_topics={"wrist":"/realsense/camera/color/image_raw", "wrist_back":"/arducam/color/image_raw"}, 
                 indy_ip ="10.208.112.143",
                 ckpt_dir = '/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/act/ckpt_dir/real_peg_cart_gains',
                 ckpt_name = 'policy_best.ckpt',
                 policy_class = 'ACT',
                 onscreen_render = False,
                 task_name = 'real_peg_cart_gains',
                 lr = 1e-4,
                 chunk_size = 100,
                 kl_weight = 10,
                 hidden_dim = 512,
                 dim_feedforward = 3200,
                 lr_backbone = 1e-5,
                 backbone = 'resnet18',
                 camera_names = ['wrist', 'wrist_back'],
                 num_epochs = 8000,
                 episode_len = 5000,
                 seed = 0,
                 temporal_agg = True,
                 is_sim = False,
                 ):
        super().__init__('act_inference')

        self.camera_topics = camera_topics 
        self.indy_ip = indy_ip


        ##########NOTE(JS): 
        self.indy = IndyDCP3(indy_ip)

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
        self.obs_ft = None
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
        
        # self.img_queues = {camera_id:deque() for camera_id in self.camera_topics.keys()}
        # self.pose_sub = self.create_subscription(Twist, "/indy/pose", self.get_pose, 10, callback_group=ReentrantCallbackGroup())
        # self.ft_sub = self.create_subscription(Twist, "/indy/ft", self.get_ft, 10, callback_group=ReentrantCallbackGroup())
        self.state_sub = self.create_subscription(RoboData, "/indy/state", self.get_state, 10, callback_group=ReentrantCallbackGroup())
        self.act_pub = self.create_publisher(CompAct, "/indy/act2", 10, callback_group=ReentrantCallbackGroup())
        # self.pose_queue = deque()
        # self.obs_pose = None
        self.bridge = cvb.CvBridge()

    def get_state(self, msg): 
        pose_msg = msg.ppos
        ft_msg = msg.wrench 
        self.obs_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
        self.obs_ft = np.array([ft_msg.linear.x, ft_msg.linear.y, ft_msg.linear.z, ft_msg.angular.x, ft_msg.angular.y, ft_msg.angular.z])
        if self.counter == 0:
            self.init_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
            self.obs_pose = np.zeros((6,))
            self.counter += 1

    def get_ft(self, msg): 
        ft_arr = np.array([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z])
        self.obs_ft = ft_arr

    def get_pose(self, msg): 
        pose_msg = msg 
        self.obs_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
        
        if self.counter == 0:
            self.init_pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
            self.obs_pose = np.zeros((6,))
            self.counter += 1
        
            
    def img_cb(self, msg, camera_id):
        curr_images = []
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # for cam_name in self.camera_names:
              # Using `image`, not `obs['images']`
        curr_image = cv2.resize(img, (640, 480))
        ##one sec
        # plt.imshow(curr_image)
        # plt.show()
        curr_image = rearrange(curr_image, 'h w c -> c h w')
        curr_images.append(curr_image)

        # curr_image = np.array(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        self.obs_img[camera_id]= curr_image
        
    def arr2msg(self, act, type="diff"):
        if type == "diff":
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
        
        elif type == "compact": 
            msg = CompAct()
            pose = act[:6]
            gains = act[6:]
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
        obs['ppos'] = np.concatenate((self.obs_pose, self.obs_ft), axis=0)
        # obs['ppos'] = self.obs_pose
        obs['pvel'] = None
        obs['effort'] = None
        # Stack images from wrist and wrist_back
        # print(self.obs_img['wrist'].shape, self.obs_img['wrist_back'].shape)
        obs['images'] = torch.cat([self.obs_img[camera_name] for camera_name in self.camera_names], dim=0).unsqueeze(0)
        # print(f"obs img shape: {obs['images'].shape}")

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
        # ckpt_dir = self.config['ckpt_dir']
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

        # load policy and stats

        # self.stats['ppos_mean'] = np.array([566.92, 220.23, 409.39987, 134.16035, 0.6010875,  84.35676])
        # self.stats['action_mean'] = np.array([566.92, 220.23, 4.0932898e+02,  1.7998888e+02, -1.0614256e-02,  8.2554703e+01])
        
        
        #print(f"PPOS mean: {self.stats['ppos_mean']}")
        #print(f"Action mean: {self.stats['action_mean']}")
        # print("stats values:", self.stats['qpos_mean'], self.stats['qpos_std'], self.stats['action_mean'], self.stats['action_std'])
        # pre_process = lambda s_ppos: (s_ppos - self.stats['qpos_mean']) / self.stats['qpos_std'] 
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
            # for t in range(max_timesteps): #NOTE(SL): To be removed
            t = self.t
            print(f"t: {t}")

            ### process previous timestep to get ppos and image_list
            # obs = ts.observation # get image and ppos
            obs = self.get_obs()

            print(f"Obs pose:{obs['ppos']}")
            current_pose_cart = copy.deepcopy(obs['ppos'])
            processed_obs = self.preprocess_obs(obs['ppos'], type="cartesian", locality="global")
            # print(f"obs img shape: {obs['images'].shape}")

            ppos_numpy = np.array(processed_obs)
            ##########NOTE(JS): Hardcoded ########
            pose = self.indy.get_control_data()['p']
            ft = self.indy.get_ft_sensor_data()
            ppos_numpy = np.array([pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], 
                                      ft['ft_Fx'], ft['ft_Fy'], ft['ft_Fz'], ft['ft_Tx'], ft['ft_Ty'], ft['ft_Tz']])
            ######################################
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

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            # action = self.postprocess_actions(action, current_pose_cart, type="cartesian", locality="", debug=False)
            pred_time = time.time()

            ### step the environment
            # ts = env.step(target_ppos) # move to target
            # print("Inference speed:", pred_time - init_time)
            # print("Target ppos:", target_ppos+ np.array(self.init_pose))
            # target_ppos2 = target_ppos + np.array(self.init_pose)
            # target_ppos2[3:] = np.array([180, 0, 90])
            # self.act_pub.publish(self.arr2msg(target_ppos))
            self.postprocess_actions(action, current_pose_cart, type="cartesian", locality="global", debug=False)
            self.t += 1

    def postprocess_actions(self, actions, current_pose, type="geometric",locality="global",debug=False): 
        if type == "geometric": 
            print(f"Actions: {actions}")
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

            # cart_command[3:6] = np.array([180, 0, 90])
            # ang2 = actions[3:]
            # ang2[0] = 180
            # rot = R.from_euler( 'xyz',ang2, degrees=True)
            # rotmat = rot.as_matrix()
            # print(f"rotmat action: {rotmat}")
        if debug: 
            print(f"command: {cart_command}, current_type: {type}, locality: {locality}")
        else:
            print(f"Non debug command: {cart_command}")
            self.act_pub.publish(self.arr2msg(cart_command, type="compact"))

    def preprocess_obs(self, obs_pose, type="geometric", locality="global"): 
        if type == "geometric":
            init_g = cart2se3(self.init_pose)
            obs_g = cart2se3(obs_pose)
            twist = get_rel_twist(init_g, obs_g)
            return twist
        elif type == "cartesian": 
            ang = obs_pose[3:6]
            rot = R.from_euler( 'xyz',ang, degrees=True)
            rotmat = rot.as_matrix()
            print(f"process obs || obs_pose: {obs_pose}, type={type}, locality={locality}")
            if locality == "init_pose" : 
                obs_pose_p = np.linalg.inv(R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix()) @ (obs_pose[:3] - self.init_pose[:3])
                obs_pose_a_rot = np.linalg.inv(R.from_euler('xyz', self.init_pose[3:6], degrees=True).as_matrix()) @ R.from_euler('xyz', obs_pose[3:6], degrees=True).as_matrix()
                obs_pose_a_euler = R.from_matrix(obs_pose_a_rot).as_euler('xyz', degrees=True)
                obs_pose = np.concatenate((obs_pose_p, obs_pose_a_euler, obs_pose[6:]), axis=0)

            return obs_pose

def main(): 
    rclpy.init()
    act_inference = ACTInference()
    rclpy.spin(act_inference)

if __name__ == "__main__": 
    main()
