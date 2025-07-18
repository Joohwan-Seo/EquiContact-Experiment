import time
from multiprocessing.managers import SharedMemoryManager
import click
import numpy as np
import torch
import dill
#ros and robot imports
import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from camera_interfaces.msg import DiffAction
#import neuromeka as nm
import cv_bridge as cvb
import cv2
import argparse
from functools import partial
from sensor_msgs.msg import Image
from collections import deque
# Diffusion Policy 
from geometric_func import cart2se3, get_rel_twist, se3_logmap, so3_logmap
import hydra
import pathlib
import skvideo.io
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


class DiffPolicyInference(Node):

    def __init__(self, ckpt_path,camera_topics={"realsense":"/realsense/camera/color/image_raw"}, indy_ip ="10.208.112.115"): 
        super().__init__('diff_policy_inference')
        self.ckpt_path = ckpt_path
        self.camera_topics = camera_topics 
        self.indy_ip = indy_ip
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        ckpt_path = "diff_policy.ckpt"
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # hacks for method-specific setup.
        self.action_offset = 0
        self.delta_action = False

        self.policy: BaseImagePolicy
        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model

        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)

        # set inference params
        self.policy.num_inference_steps = 16 # DDIM inference iterations
        self.policy.n_action_steps = self.policy.horizon - self.policy.n_obs_steps + 1


        # setup experiment
        self.frequency = 5
        self.dt = 1/self.frequency

        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.n_obs_steps = self.cfg.n_obs_steps
        steps_per_inference = 6 
        action_offset = np.zeros((6, ))
        print("n_obs_steps: ", self.n_obs_steps)
        print("steps_per_inference:", steps_per_inference)
        print("action_offset:", action_offset)
        self.initialize_sensors()
        time.sleep(5)
        self.inference_loop = self.create_timer(self.dt, self.inference)
        # self.create_timer(self.dt, self.sensor_loops_test)

    def initialize_sensors(self): 
        #self.indy = nm.IndyDCPClient(self.indy_ip)
        self.img_subs = {camera_id: self.create_subscription(Image, topic, partial(self.img_cb, camera_id=camera_id), 10, callback_group=ReentrantCallbackGroup()) for camera_id, topic in self.camera_topics.items()}
        self.img_queues = {camera_id:deque() for camera_id in self.camera_topics.keys()}
        self.pose_sub = self.create_subscription(Twist, "/indy/pose", self.get_pose, 10, callback_group=ReentrantCallbackGroup())
        self.act_pub = self.create_publisher(DiffAction, "/indy/act", 10, callback_group=ReentrantCallbackGroup())
        self.pose_queue = deque()
        self.bridge = cvb.CvBridge()

    def get_pose(self, msg): 
        #pose = self.indy.get_control_data()['p']
        pose_msg = msg 
        pose = np.array([pose_msg.linear.x, pose_msg.linear.y, pose_msg.linear.z, pose_msg.angular.x, pose_msg.angular.y, pose_msg.angular.z])
        if len(self.pose_queue) >= self.n_obs_steps: 
            self.pose_queue.popleft()
        self.pose_queue.append(pose)
    
    def img_cb(self, msg, camera_id):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        queue_size = len(self.img_queues[camera_id])
        if queue_size >= self.n_obs_steps: 
            self.img_queues[camera_id].popleft()
        self.img_queues[camera_id].append(img)
        #self.get_logger().info(f"Received image from {camera_id}")

    def pack_obs(self): 
        obs_dict = {}
        poses = np.stack(self.pose_queue, axis=0)
        imgs = np.stack(self.img_queues["realsense"], axis=0)
        obs_dict["agent_pos"] = poses 
        obs_dict["image"] = imgs
        obs_dict_np = get_real_obs_dict(env_obs = obs_dict, shape_meta = self.cfg.task.shape_meta)
        obs_dict_torch = dict_apply(obs_dict_np, 
        lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict_torch
    
    def inference(self, timeout=30, latency=0.5): 
        
        with torch.no_grad():
            s = time.time()
            print(f"Pose queue: {len(self.pose_queue)}")
            print(f"Image queue: {len(self.img_queues['realsense'])}")

            if len(self.pose_queue) == self.n_obs_steps and len(self.img_queues["realsense"]) == self.n_obs_steps:
                obs_dict_torch = self.pack_obs()
                print(f"Img shape: {obs_dict_torch['image'].shape}, Pose shape: {obs_dict_torch['agent_pos'].shape}")
                result = self.policy.predict_action(obs_dict_torch)
                # this action starts from the first obs step
                action = result['action'][0].detach().to('cpu').numpy()
                print(f"Action: {action}")
                diffaction = self.arr2msg(action)
                self.act_pub.publish(diffaction)
                #print('Inference latency:', time.time() - s)
   
    def arr2msg(self, act):
        msg = DiffAction()
        twists = msg.actions
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
        
def test_sensor_loops(): 
    rclpy.init()
    diff_pipeline = DiffPolicyInference(ckpt_path = "/home/exx/diff_policy/data/outputs/2025.02.26/18.21.29_train_diffusion_unet_image_peg_hole_image/checkpoints/latest.ckpt")
    rclpy.spin(diff_pipeline)

if __name__ == "__main__": 
    test_sensor_loops()
