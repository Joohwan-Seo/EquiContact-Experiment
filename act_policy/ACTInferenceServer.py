import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from camera_interfaces.msg import DiffAction, CompAct, RoboData, ActActivity
from camera_interfaces.srv import ACTInference
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
from einops import rearrange

from act.utils import set_seed
from act.policy import ACTPolicy, CNNMLPPolicy

from act.sim_env import BOX_POSE
import IPython
import collections
import json
import cv2

class ACTInferenceServer(Node):

    def __init__(self, pick_config, place_config, debug=False ): 
        super().__init__('act_inference_service')
        self.pick_config = pick_config 
        self.place_config = place_config
        
        self.debug = False
               
        self.msg_types = {"Image": Image, "Twist": Twist, "DiffAction": DiffAction, "CompACT": CompAct, "RoboData": RoboData}
        self.pubs = {}

        self.t = 0
        self.obs = {}

        self.state_cb_time_prev = time.time()

        # initialization parameters for the cameras; assuming that pick and place have the same camera names
        self.camera_names = self.pick_config['task_config']['camera_names']
        self.crop_image = False
        self.mask_image = False

        self.pre_pick_move_finished_flag = False
        self.pick_finished_flag = False
        self.grasp_move_finished_flag = False
        self.place_finished_flag = False

        self.state_dim = {}
        self.action_dim = {}
        self.num_queries = {}
        self.episode_len = {}
        self.state_category = {}
        self.action_category = {}

        self.pre_process = {}
        self.post_process = {}

        self.wp0_pre_arrived = False
        self.wp0_arrived = False
        self.wp1_arrived = False
        self.wp2_arrived = False
        self.wp3_arrived = False

        self.wp0_count = 0

        self.stats = {}
        
        self.init_sensors()

        self.service = self.create_service(ACTInference, 'act_inference', self.service_cb, callback_group=ReentrantCallbackGroup())
        
        self.active = False
        
        self.counter = 0

        self.last_pose = np.array([457.2, -304.8, 280, 180, 0, 90])
        self.last_gains = np.ones((6,)) * 1000

        self.current_time = time.time()
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

            self.create_subscription(Image, topic, partial(self.img_cb, camera_id=cam_name), 20, callback_group=ReentrantCallbackGroup())

        # intiailize proprioceptive sensors
        self.create_subscription(RoboData, "/indy/state", self.state_cb, 20, callback_group=ReentrantCallbackGroup())
        ## initialize binary subscriber 
        # self.create_subscription(ActActivity, "act_activity", self.activity_cb, 10, callback_group=ReentrantCallbackGroup())
        # initialize publishers
        self.ft_rebias_pub = self.create_publisher(Bool, "/indy/rebias", 10, callback_group=ReentrantCallbackGroup())
        self.pubs['action'] = self.create_publisher(CompAct, "/indy/act2", 20)
    
    def make_policy(self, policy_class, policy_config):
        if policy_class == 'ACT':
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError
        return policy
    
    def init_policy(self, task_config, eval_config, train_config, policy_config, task): 
        dt = 1/eval_config['frequency'] # 1/30
        ckpt_path = os.path.join(eval_config['ckpt_dir'], eval_config['ckpt_name'])         
        policy = self.make_policy("ACT", policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
        print(loading_status)
        policy.cuda()
        policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(eval_config['ckpt_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats[task] = pickle.load(f)

        print(f"current task: {task}, state_dim: {self.stats[task]['state_mean'].shape}, action_dim: {self.stats[task]['action_mean'].shape}")
        self.pre_process[task] = lambda s_ppos: (s_ppos - self.stats[task]['state_mean']) / self.stats[task]['state_std'] 
        self.post_process[task] = lambda a: a * self.stats[task]['action_std'] + self.stats[task]['action_mean']

        return dt, policy
    
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
        
        # print(f"state_cb_time: {time.time() - self.state_cb_time_prev}")
        self.state_cb_time_prev = time.time()
        
    # def activity_cb(self, msg): 
    #     self.active = msg.active
    #     self.reference_frame = np.array(msg.reference_frame)
    #     task = msg.task

    #     if task == self.current_task:
    #         return
        
    #     self.unload_model()

    #     self.load_model(task)

    #     self.current_task = task

    #     print(self.active)

    def load_model(self, task_config, train_config, eval_config, task):
        self.state_dim[task] = task_config['state_dim']
        self.action_dim[task] = task_config['action_dim']
        self.num_queries[task] = task_config['num_queries']
        self.state_category[task] = task_config['state_category']
        self.action_category[task] = task_config['action_category']
        policy_class = task_config['policy_class']
        camera_names = task_config['camera_names']
        crop_image = train_config['crop_image']
        crop_ratio = train_config['crop_ratio']
        mask_image = train_config['mask_image']
        mask_ratio = train_config['mask_ratio']
        self.episode_len[task] = eval_config['episode_len']
        
        self.temporal_agg = eval_config['temporal_agg']

        # fixed policy config
        
        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': float(train_config['lr']),
                        'num_queries': self.num_queries[task],
                        'kl_weight': train_config['kl_weight'],
                        'hidden_dim':task_config['hidden_dim'],
                        'dim_feedforward':task_config['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': self.camera_names,
                        'state_dim': self.state_dim[task],
                        'action_dim':self.action_dim[task],
                        }

        dt, policy = self.init_policy(task_config, eval_config, train_config, policy_config, task)

        buffer = 100

        return dt, policy
    
    def reset_all_time_arrays(self, task):
        buffer = 100
        self.all_time_poses_np = np.zeros((self.episode_len[task] + buffer, self.episode_len[task] + buffer + self.num_queries[task],4,4))
        self.all_time_gripper_np = np.zeros((self.episode_len[task] + buffer, self.episode_len[task] + buffer + self.num_queries[task], 1))
        self.all_time_gains_np = np.zeros((self.episode_len[task] + buffer, self.episode_len[task] + buffer + self.num_queries[task], 6))


    def load_pick_place(self): 
        pick_task_config = self.pick_config['task_config']
        pick_eval_config = self.pick_config['eval_config']
        pick_train_config = self.pick_config['train_config']
        self.pick_dt, self.pick_policy = self.load_model(pick_task_config, pick_train_config, pick_eval_config, 'pick')
        place_task_config = self.place_config['task_config']
        place_eval_config = self.place_config['eval_config']
        place_train_config = self.place_config['train_config']
        self.place_dt, self.place_policy = self.load_model(place_task_config, place_train_config, place_eval_config, 'place')
    
    # def unload_model(self):
    #     if self.policy is not None:
    #         del self.policy
    #         self.policy = None
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #         gc.collect()
    #         self.get_logger().info("Unloaded previous model and cleared CUDA cache")   

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
            # print(f"Non-debug Action: {pose}, {gains}, {gripper}")
            # print(f"current pose: {self.obs['ppos']}")
            # print(f"Non-debug Action: {gripper}")
            self.pubs['action'].publish(msg)
            # self.pubs['action'].publish(msg)
    
    def process_actions(self, actions, task):
        all_action_poses_np = np.zeros((self.num_queries[task], 4, 4))
        all_action_gains_np = np.ones((self.num_queries[task], 6)) * 1000 # default gains

        
        for action_type in self.action_category[task]:
            if action_type == "world_pose":
                all_action_poses_np = pose_data_to_hom_matrix(actions[:, 0:6])
            
            elif action_type == "relative_pose":
                all_action_poses_np = np.expand_dims(self.current_pose, axis = 0) @ batch_pose_to_hom_matrices(actions[:, 0:6])

            elif action_type == "gains":
                all_action_gains_np = np.power(10, actions[:, 6:])
            
            elif action_type == "gripper":
                all_gripper_actions_np = actions[:, -1].reshape((-1,1))

        if task == 'place':
            all_gripper_actions_np = np.ones((self.num_queries[task], 1)) # Gripper Close
        elif task == 'pick':
            all_gripper_actions_np = np.ones((self.num_queries[task], 1))*0.01 # Gripper Open

        return all_action_poses_np, all_action_gains_np, all_gripper_actions_np

    def format_obs(self, task): 
        # Process states and images
        # from numpy to torch

        obs = collections.OrderedDict()
        self.current_pose = command_to_hom_matrix(self.obs['ppos'])
        states = []
        for state_type in self.state_category[task]:
            if state_type == "world_pose":
                state = command_to_pose_data(self.obs['ppos'])

            elif state_type == "GCEV":
                gd = command_to_hom_matrix(self.reference_frame[task])
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
        assert states.shape[0] == self.state_dim[task], f"State dimension mismatch: {states.shape[0]} != {self.state_dim[task]}"

        normalized_states = self.pre_process[task](states)

        # Process images
        images = []
        for camera_name in self.camera_names:
            images.append(self.obs[camera_name])
        
        images = np.stack(images, axis=0)

        obs['state'] = torch.from_numpy(normalized_states).float().cuda().unsqueeze(0)
        obs['images'] = torch.from_numpy(images / 255.0).float().cuda().unsqueeze(0)
        return obs 
    
    def service_cb(self, request, response): 
        pick_reference_frame  = request.pick_reference_frame 
        place_reference_frame = request.place_reference_frame
        self.reference_frame = {}
        self.reference_frame['pick'] = np.array(pick_reference_frame)
        self.reference_frame['place'] = np.array(place_reference_frame)
        self.get_logger().info(f"Starting full task")
        self.load_pick_place()
        self.reset_all_time_arrays('pick')
        

        # self.pick_loop = self.create_timer(self.pick_dt, self.pick, callback_group=ReentrantCallbackGroup())
        # self.move_loop = self.create_timer(self.pick_dt, self.move, callback_group=ReentrantCallbackGroup())
        # self.place_loop = self.create_timer(self.place_dt, self.place, callback_group=ReentrantCallbackGroup())

        self.pipeline_loop = self.create_timer(self.pick_dt, self.pipeline, callback_group=ReentrantCallbackGroup())

        response.finished = True

        return response
    
    def ACTinference(self, task):
        # self.get_logger().info(f"In {task} task")
        if self.temporal_agg: 
            query_freq = 1
        else:
            query_freq = self.num_queries[task]

        # print("dt in the inference loop", time.time() - self.current_time)

        t = self.t

        command_pose = self.reference_frame[task]
        gains = np.ones((6,)) * 1000

        if t > self.episode_len[task]:
            # stop further timer callbacks
            if task == "pick":
                self.get_logger().info("Pick task finished")
                self.pick_finished_flag= True
                self.t = 0

            elif task == "place":
                self.get_logger().info("Place task finished")
                self.place_finished_flag = True
                self.t = 0
                self.pipeline_loop.cancel()

                return

        with torch.inference_mode():
            obs = self.format_obs(task)

            curr_image = obs['images']
            state_in = obs['state']

            # print(f"current_state: {state_in}")

            
            if t % query_freq == 0:
                if task == 'pick':
                    all_actions = self.pick_policy(state_in, curr_image).squeeze(0).cpu().numpy()
                elif task == 'place':
                    all_actions = self.place_policy(state_in, curr_image).squeeze(0).cpu().numpy()
                unnormalized_actions = self.post_process[task](all_actions)
                all_action_poses_np, all_action_gains_np, all_gripper_actions_np = self.process_actions(unnormalized_actions, task) # updating all_time_poses_np and all_time_gains_np
                # print(t, all_actions[0])

            if self.temporal_agg:
                self.all_time_poses_np[[t], t:t + self.num_queries[task], :, :] = all_action_poses_np
                poses_for_curr_step = self.all_time_poses_np[:, t]
                poses_populated = (np.linalg.det(poses_for_curr_step[:,:3,:3]) != 0)
                            # print(actions_for_curr_step.shape)
                poses_for_curr_step = poses_for_curr_step[poses_populated]

                self.all_time_gains_np[[t], t:t + self.num_queries[task], :] = all_action_gains_np
                gains_for_curr_step = self.all_time_gains_np[:, t]
                gains_populated = np.all(gains_for_curr_step != 0, axis = 1)
                gains_for_curr_step = gains_for_curr_step[gains_populated]

                self.all_time_gripper_np[[t], t:t + self.num_queries[task], :] = all_gripper_actions_np
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

            
        # convert translation, rotation and gains to the command message format
        g = np.eye(4)
        g[:3, 3] = translation
        g[:3, :3] = rotation
        command_pose = hom_matrix_to_command(g) # in mm and degrees
        self.shoot_action(command_pose, gains, gripper, debug = self.debug)

        self.last_pose = command_pose
        self.last_gains = gains

        self.t += 1

    
    def pipeline(self):
        if not self.pre_pick_move_finished_flag:
            gains = np.ones((6,)) * 1000
            command_pose_wp0 = self.reference_frame['pick'] + np.array([0, 0, 80, 0, 0, 0])

            if not self.wp0_pre_arrived:
                self.shoot_action(command_pose_wp0, gains, 0)
                
                if np.linalg.norm(self.obs['ppos'][:3] - command_pose_wp0[:3]) < 3:
                    self.wp0_pre_arrived = True
                    self.pre_pick_move_finished_flag = True
                    self.get_logger().info("Pre-pick move finished, moving to pick task")

        elif not self.pick_finished_flag:
            self.ACTinference('pick')

        elif self.pick_finished_flag and not self.grasp_move_finished_flag:
            gains = np.ones((6,)) * 1000
            command_pose = self.last_pose

            command_pose_wp0 = command_pose.copy()
            command_pose_wp1 = np.array([command_pose[0], command_pose[1], command_pose[2] + 150, 180, 0, 90])
            command_pose_wp2 = np.array([500, 0, 500, 180, 0, 90])

            # process command_pose_wp3
            R_init = R.from_euler('xyz', self.reference_frame['place'][3:], degrees=True).as_matrix()
            command_pose_wp3 = self.reference_frame['place'].copy()
            command_pose_wp3[:3] = command_pose_wp3[:3] + R_init @ np.array([0, 0, -80])


            current_pose = self.obs['ppos']

            if not self.wp0_arrived:
                if self.wp0_count < 60:
                    print(command_pose_wp0)
                    self.shoot_action(command_pose_wp0, gains, 1)
                    self.wp0_count += 1
                else:
                    self.wp0_arrived = True
                    self.get_logger().info("Grapsed the peg")

            else:
                if not self.wp1_arrived:
                    self.shoot_action(command_pose_wp1, gains, 1)

                    if np.linalg.norm(current_pose[:3] - command_pose_wp1[:3]) < 10:
                        self.wp1_arrived = True
                        self.get_logger().info("Waypoint 1 arrived")

                else:
                    if not self.wp2_arrived:
                        
                        self.shoot_action(command_pose_wp2, gains, 1)

                        if np.linalg.norm(current_pose[:3] - command_pose_wp2[:3]) < 10:
                            self.wp2_arrived = True
                            self.get_logger().info("Waypoint 2 arrived")
                    else:
                        if not self.wp3_arrived:
                            self.shoot_action(command_pose_wp3, gains, 1)

                            if np.linalg.norm(current_pose[:3] - command_pose_wp3[:3]) < 10:
                                self.wp3_arrived = True
                                self.grasp_move_finished_flag = True
                                self.get_logger().info("Grasp move finished, moving to place task")
                                self.reset_all_time_arrays('place')
                                rebias_msg = Bool()
                                rebias_msg.data = True
                                self.get_logger().info("Rebiasing the FT Sensor")
                                self.ft_rebias_pub.publish(rebias_msg)
                                time.sleep(2) # rebias the FT sensor

        elif self.grasp_move_finished_flag and not self.place_finished_flag:
            self.ACTinference('place')


        


        
def main(args): 
    debug = args.debug 
    config_path_place = args.config_place 
    config_path_pick = args.config_pick
    rclpy.init(args=None)
    config_place = yaml.load(open(config_path_place, "r"), Loader=yaml.Loader)
    config_pick = yaml.load(open(config_path_pick, "r"), Loader=yaml.Loader)
    config = {}

    for task in ['pick', 'place']:
        if task == 'pick':
            config[task] = config_pick
        elif task == 'place':
            config[task] = config_place
        else:
            raise NotImplementedError

    act_inference = ACTInferenceServer(config_pick, config_place, debug=debug)
    try:
        rclpy.spin(act_inference)
    finally:
        act_inference.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_place', type=str, 
                        default='/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pattern_platform_05_with_distractor.yaml', 
                        help='Path to the config file')
    parser.add_argument('--config_pick', type=str, 
                        default='/home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pick_02.yaml', 
                        help='Path to the config file')
    parser.add_argument('--debug', type= bool, default=False, help='Debug mode')
    args = parser.parse_args()
    main(args)


