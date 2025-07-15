
import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from camera_interfaces.msg import DiffAction,CompAct, RoboData, ActActivity
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

    def __init__(self, config, reference_frame = None, debug=False ): 

        super().__init__('act_inference_node')
        self.config = config 
        self.debug = False
        self.task_config = config['task_config']
        self.eval_config = config['eval_config']
        self.train_config = config['train_config']

        self.state_dim = self.task_config['state_dim']
        self.action_dim = self.task_config['action_dim']
        self.num_queries = self.task_config['num_queries']
        self.state_category = self.task_config['state_category']
        self.action_category = self.task_config['action_category']
        self.policy_class = self.task_config['policy_class']

        self.camera_names = self.task_config['camera_names']

        self.crop_image = self.train_config['crop_image']
        self.crop_ratio = self.train_config['crop_ratio']
        self.mask_image = self.train_config['mask_image']
        self.mask_ratio = self.train_config['mask_ratio']

        self.rot6d = self.task_config['rot6d']

        ##### NOTE(JS): Later to be changed using the response from service ######
        if reference_frame is None:
            self.reference_frame = np.array(self.train_config['reference_frame'])
        else:
            self.reference_frame = np.array(reference_frame)

        print(self.reference_frame)
        ##########################################################################
       
        self.msg_types = {"Image": Image, "Twist": Twist, "DiffAction": DiffAction, "CompACT": CompAct, "RoboData": RoboData}
        self.pubs = {}

        dt = 1/self.eval_config['frequency']
        self.episode_len = self.eval_config['episode_len']
        self.temporal_agg = self.eval_config['temporal_agg']

        # fixed policy config
        if self.policy_class == "ACT":
            lr_backbone = 1e-5
            backbone = 'resnet18'
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            self.policy_config = {'lr': float(self.train_config['lr']),
                            'num_queries': self.num_queries,
                            'kl_weight': self.train_config['kl_weight'],
                            'hidden_dim': self.task_config['hidden_dim'],
                            'dim_feedforward': self.task_config['dim_feedforward'],
                            'lr_backbone': lr_backbone,
                            'backbone': backbone,
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': self.camera_names,
                            'state_dim': self.state_dim,
                            'action_dim': self.action_dim,
                            }
        else:
            raise NotImplementedError
        
        self.t = 0
        self.obs = {}
        
        self.init_sensors()
        self.init_policy()

        # print(self.obs)

        if self.temporal_agg:
            # Initialize all_time_actions if it's the first call
            self.all_time_poses_np = np.zeros((self.episode_len, self.episode_len + self.num_queries,4,4))
            self.all_time_gripper_np = np.zeros((self.episode_len, self.episode_len + self.num_queries, 1))
            self.all_time_gains_np = np.zeros((self.episode_len, self.episode_len + self.num_queries, 6))

            # if "gains" in self.action_category:
                # pass

        self.exec_loop = self.create_timer(dt, self.inference, callback_group=ReentrantCallbackGroup())
        self.active = True

        self.current_time = time.time()
        
        self.counter = 0

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

            self.create_subscription(Image, topic, partial(self.img_cb, camera_id=cam_name), 10)

        # intiailize proprioceptive sensors
        self.create_subscription(RoboData, "/indy/state", self.state_cb, 10)
        ## initialize binary subscriber 
        self.create_subscription(ActActivity, "act_activity", self.activity_cb, 10)
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

    def init_policy(self): #NOTE(JS): Done
        ckpt_path = os.path.join(self.eval_config['ckpt_dir'], self.eval_config['ckpt_name']) 
        self.policy = self.make_policy(self.policy_class, self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(self.eval_config['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.pre_process = lambda s_ppos: (s_ppos - self.stats['state_mean']) / self.stats['state_std'] 
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

    def img_cb(self, msg, camera_id): #NOTE(JS): Done

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
        # TODO(JS) Rewrite
        self.obs["ppos"] = np.array([msg.ppos.linear.x, msg.ppos.linear.y, msg.ppos.linear.z,
                                     msg.ppos.angular.x, msg.ppos.angular.y, msg.ppos.angular.z], dtype=np.float32)
        self.obs["pvel"] = np.array([msg.pvel.linear.x, msg.pvel.linear.y, msg.pvel.linear.z,
                                     msg.pvel.angular.x, msg.pvel.angular.y, msg.pvel.angular.z], dtype=np.float32)
        self.obs["FT"] = np.array([msg.ft.linear.x, msg.ft.linear.y, msg.ft.linear.z,
                                       msg.ft.angular.x, msg.ft.angular.y, msg.ft.angular.z], dtype=np.float32)
        
    def activity_cb(self, msg): 
        self.active = msg.active
        self.reference_frame = np.array(msg.reference_frame)
        print(self.active)   

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

        # if gripper >= 0.9:
        if gripper >= 0.3:
            gripper_value = int(1)
        else:
            gripper_value = int(0)

        msg.gripper_state = gripper_value
        if debug:
            print(f"Action: {pose}, {gains}, {gripper}")
        else:
            # print(f"Non-debug Action: {pose}, {gains}, {gripper}")
            # print(f"Current pose: {self.obs['ppos']}")
            # print(f"Non-debug Action: {gripper}")
            self.pubs['action'].publish(msg)
    
    def process_actions(self, actions):
        all_action_poses_np = np.zeros((self.num_queries, 4, 4))
        all_action_gains_np = np.ones((self.num_queries, 6)) * 1000 # default gains
        all_gripper_actions_np = np.ones((self.num_queries, 1)) # default gripper

        for action_type in self.action_category:
            if action_type == "world_pose":
                # self.all_time_poses_np[self.t, self.t:self.t+self.num_queries, :, :] = pose_data_to_hom_matrix(actions[:, 0:6])
                if self.rot6d:
                    all_action_poses_np = batch_rot6d_pose_to_hom_matrices(actions[:, 0:9])
                else:
                    all_action_poses_np = batch_pose_to_hom_matrices(actions[:, 0:6])                
            
            elif action_type == "relative_pose":
                # self.all_time_poses_np[self.t, self.t:self.t+self.num_queries, :, :] = self.current_pose.unsqueeze(0) @ batch_pose_to_hom_matrices(actions[:, 0:6])
                all_action_poses_np = np.expand_dims(self.current_pose, axis = 0) @ batch_pose_to_hom_matrices(actions[:, 0:6])

            elif action_type == "gains":
                # self.all_time_gains_np[self.t, self.t:self.t+self.num_queries, :] = np.power(10, actions[:, 6:])
                if self.rot6d:
                    all_action_gains_np = np.power(10, actions[:, 9:15])                
                else:
                    all_action_gains_np = np.power(10, actions[:, 6:])
            
            elif action_type == "gripper":
                all_gripper_actions_np = actions[:, -1].reshape((-1,1))

        return all_action_poses_np, all_action_gains_np, all_gripper_actions_np

    def format_obs(self): 
        # Process states and images
        # from numpy to torch

        obs = collections.OrderedDict()
        self.current_pose = command_to_hom_matrix(self.obs['ppos'])
        states = []
        for state_type in self.state_category:
            if state_type == "world_pose":
                if self.rot6d:
                    state = command_to_rot6d_pose_data(self.obs['ppos'])
                else:
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

        # print(f"Current state: {states}")

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

        # print(time.time() - self.current_time)
        self.current_time = time.time()

        t = self.t
        if self.t >= self.episode_len:
            print("Reached maximum timesteps")
            self.exec_loop.cancel()
            raise KeyboardInterrupt("Reached maximum timesteps")
            return
        if not self.active: 
            return None
        
        with torch.inference_mode():
            obs = self.format_obs()

            curr_image = obs['images']
            state_in = obs['state']

            if self.policy_class == "ACT":
                if t % query_freq == 0:
                    all_actions = self.policy(state_in, curr_image).squeeze(0).cpu().numpy()
                    unnormalized_actions = self.post_process(all_actions)
                    all_action_poses_np, all_action_gains_np, all_gripper_actions_np = self.process_actions(unnormalized_actions) # updating all_time_poses_np and all_time_gains_np

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

        self.t += 1

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
        
def main(args): 
    debug = True
    config_path = args.config 
    rclpy.init(args=None)
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    act_inference = ACTInference(config, reference_frame=args.reference_frame, debug=debug)
    rclpy.spin(act_inference)

if __name__ == '__main__':
    import argparse
    from pipeline.diffusion_edf.FullPipelineClient import FullPipelineClient 


    def move_to_free(client, pose, gripper_state=0): 
        act_msg = CompAct()
        act_msg.cart_pose.linear.x = pose[0]
        act_msg.cart_pose.linear.y = pose[1]
        act_msg.cart_pose.linear.z = pose[2] 
        act_msg.cart_pose.angular.x = pose[3]
        act_msg.cart_pose.angular.y = pose[4]
        act_msg.cart_pose.angular.z = pose[5]
        act_msg.ad_gains.linear.x = 1000.0
        act_msg.ad_gains.linear.y = 1000.0
        act_msg.ad_gains.linear.z = 1000.0
        act_msg.ad_gains.angular.x = 1000.0
        act_msg.ad_gains.angular.y = 1000.0
        act_msg.ad_gains.angular.z = 1000.0
        act_msg.gripper_state = gripper_state
        client.robot_pub.publish(act_msg)

    # def process_pick_edf_response(response):
    #     default_R = np.array([[0, 1, 0],
    #                           [1, 0, 0],
    #                           [0, 0, -1]])
        
    #     pick_pos = np.array(response.position).reshape((-1, 3))
    #     pick_ori = np.array(response.orientation).reshape((-1, 3))

    #     pick_rotm = R.from_euler('xyz', pick_ori, degrees = True).as_matrix()
    #     cost = np.zeros(pick_rotm.shape[0])

    #     for i in range(pick_ori.shape[0]):
    #         cost[i] = np.trace(np.eye(3) - pick_rotm[i].T @ default_R)
        
    #     idx = np.argmin(cost)

    #     return pick_pos[idx], pick_ori[idx]

    def process_pick_edf_response(response):
        pick_pos = np.array(response.position).reshape((-1, 3))
        pick_ori = np.array(response.orientation).reshape((-1, 3))

        pick_rotm = R.from_euler('xyz', pick_ori, degrees = True).as_matrix()

        tip_pos = np.zeros((pick_pos.shape[0], 3))
        for i in range(pick_pos.shape[0]):
            tip_pos[i] = pick_pos[i] + pick_rotm[i] @ np.array([0, 0, 190])

        print("tip pos's", tip_pos)
        
        for i in range(tip_pos.shape[0]):
            print(pick_rotm[i])

        tip_pos_mean = np.mean(tip_pos, axis = 0)
        tip_ori_prefix = R.from_euler('xyz', [180, 0, 90], degrees = True).as_matrix()

        place_pos_processed = tip_pos_mean + tip_ori_prefix @ np.array([0, 0, -190])
        place_ori_processed = R.from_matrix(tip_ori_prefix).as_euler('xyz', degrees = True)

        return place_pos_processed, place_ori_processed

    def process_place_edf_response(response, platform):
        place_pos = np.array(response.position).reshape((-1, 3))
        place_ori = np.array(response.orientation).reshape((-1, 3))

        place_rotm = R.from_euler('xyz', place_ori, degrees = True).as_matrix()

        if platform == 'default':
            rotm_default = R.from_euler('xyz', [180, 0, 90], degrees = True).as_matrix()
        elif platform == 'tilted':
            rotm_default = R.from_euler('xyz', [-30, 180, 0], degrees = True).as_matrix() 

        tip_pos = np.zeros((place_pos.shape[0], 3))
        for i in range(place_pos.shape[0]):
            tip_pos[i] = place_pos[i] + place_rotm[i] @ np.array([0, 0, 285])

        print("tip pos's", tip_pos)
        
        for i in range(tip_pos.shape[0]):
            print(place_rotm[i])

        tip_pos_mean = np.mean(tip_pos, axis = 0)
        tip_ori_mean = R.from_euler('xyz', place_ori, degrees = True).mean().as_matrix()

        tip_ori_list = []

        for i in range(4):
            tip_ori_list.append(tip_ori_mean @ R.from_euler("z", i * np.pi / 2).as_matrix())

        min_dist = 100
        if platform == 'default':
            for i in range(4):
                dist = np.trace(np.eye(3) - tip_ori_list[i].T @ rotm_default)
                if dist < min_dist:
                    min_dist = dist
                    tip_ori_mean = tip_ori_list[i]
        else:
            for i in range(4):
                dist = np.trace(np.eye(3) - tip_ori_list[i].T @ rotm_default)
                if dist < min_dist:
                    min_dist = dist
                    tip_ori_mean = tip_ori_list[i]

        place_pos_processed = tip_pos_mean + tip_ori_mean @ np.array([0, 0, -285])

        place_ori_processed = R.from_matrix(tip_ori_mean).as_euler('xyz', degrees = True)

        return place_pos_processed, place_ori_processed
    
    def full_inference():

        # platform = "default" # or tilted
        platform = "tilted"

        rclpy.init()
        edf_client = FullPipelineClient()
        
        # act_client = ACTInferenceClient()
        home_pose = np.array([350.0, -186.5, 522.1, 180.0, 0.0, 90.0])
        offset_length = 80.0
        move_to_free(edf_client, home_pose, gripper_state=0)
        time.sleep(5)

        act_activity = ActActivity()
        
        # #### PICK EXECUTION ####
        future = edf_client.get_edf("pick")
        edf_client.get_logger().info("Waiting for pick service")
        rclpy.spin_until_future_complete(edf_client, future)
        response = future.result()

        pick_pos, pick_ori = process_pick_edf_response(response)

        print(f"Pick pose positions:\n {pick_pos}, and orientations:\n {pick_ori}")

        time.sleep(3)

        R_init = R.from_euler('xyz', pick_ori, degrees=True).as_matrix()
        offset = R_init @ np.array([0, 0, -offset_length])
        pick_command = [pick_pos[0] + offset[0],
                        pick_pos[1] + offset[1],
                        pick_pos[2] + offset[2],
                        pick_ori[0],
                        pick_ori[1],
                        pick_ori[2]]
        
        edf_client.get_logger().info(f"Pick command: {pick_command}")
        
        move_to_free(edf_client, pick_command, gripper_state=0)
        time.sleep(7)

        pick_pose = np.concatenate((pick_pos, pick_ori)).astype(np.float64).tolist()

        # os.system(f"python /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/ACTInference_renew.py --reference_frame " + str(pick_pose[0]) 
        #           + " " + str(pick_pose[1]) + " " + str(pick_pose[2]) + " " + str(pick_pose[3]) + " " + str(pick_pose[4]) + " " + str(pick_pose[5]) +
        #           " " + "--config /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pick_02.yaml")

        # future = act_client.send_request("pick", pick_pose)
        # rclpy.spin_until_future_complete(act_client, future)
        # response = future.result()
        config_path = '/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pick_02.yaml'
        config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
        act_node = ACTInference(config = config,
                                reference_frame=pick_pose, debug=False)
        try:
            rclpy.spin(act_node)
        except KeyboardInterrupt:
            act_node.get_logger().info("Keyboard interrupt, shutting down ACTInference node.")
        time.sleep(1)

        pose_after_pick = list(np.array(act_node.obs["ppos"], dtype=np.float64))

        ### Grasp the peg
        move_to_free(edf_client, pose_after_pick, gripper_state=1)
        time.sleep(3)

        command_1 = np.array([pose_after_pick[0], pose_after_pick[1], pose_after_pick[2] + 150.0, 180.0, 0.0, 90.0], dtype = np.float64)
        act_node.destroy_node()
        del act_node
        #### MOVE TO HOME POSE ####
        move_to_free(edf_client, command_1, gripper_state=1)
        time.sleep(2)

        command_2 = np.array([500.0, 0.0, 400, 180.0, 0.0, 90.0], dtype = np.float64)

        move_to_free(edf_client, command_2, gripper_state=1)
        time.sleep(2)

        # #### PLACE EXECUTION ####
        future = edf_client.get_edf("place")
        edf_client.get_logger().info("Waiting for place service")
        rclpy.spin_until_future_complete(edf_client, future)
        response = future.result()

        place_pos, place_ori = process_place_edf_response(response, platform)
        print(f"Place pose positions:\n {place_pos}, and orientations:\n {place_ori}")

        R_init = R.from_euler('xyz', place_ori, degrees=True).as_matrix()
        offset = R_init @ np.array([0, 0, -70])
        place_command = [place_pos[0] + offset[0],
                        place_pos[1] + offset[1],
                        place_pos[2] + offset[2],
                        place_ori[0],
                        place_ori[1],
                        place_ori[2]]
        # edf_client.get_logger().info(f"Place command: {place_command}")
        move_to_free(edf_client, place_command, gripper_state=1)
        edf_client.get_logger().info(f"Published pose command for place: {place_pos}, {place_ori}")

        #kill edf_client
        edf_client.destroy_node()
        del edf_client
        
        time.sleep(5)

        place_pose = np.concatenate((place_pos, place_ori)).astype(np.float64).tolist()

        config_path = '/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pattern_platform_04.yaml'
        config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
        act_node = ACTInference(config = config,
                                reference_frame=place_pose, debug=False)
        try:
            rclpy.spin(act_node)
        except KeyboardInterrupt:
            act_node.get_logger().info("Keyboard interrupt, shutting down ACTInference node.")
            rclpy.shutdown()



    def test_edf_client():
        rclpy.init()
        edf_client = FullPipelineClient()
        # future = edf_client.get_edf("pick")
        # edf_client.get_logger().info("Waiting for pick service")
        # rclpy.spin_until_future_complete(edf_client, future)
        # response = future.result()
        
        # pick_pos, pick_ori = process_pick_edf_response(response)

        # print(f"Pick pose positions:\n {pick_pos}, and orientations:\n {pick_ori}")

        # time.sleep(3)

        platform = 'tilted'

        future = edf_client.get_edf("place")
        edf_client.get_logger().info("Waiting for place service")
        rclpy.spin_until_future_complete(edf_client, future)
        response = future.result()

        place_pos, place_ori = process_place_edf_response(response, platform)
        print(f"Place pose positions:\n {place_pos}, and orientations:\n {place_ori}")

    full_inference()