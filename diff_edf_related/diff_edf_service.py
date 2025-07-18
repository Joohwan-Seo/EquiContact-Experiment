import rclpy 
import open3d as o3d
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField

from camera_interfaces.srv import EDF 

import argparse
import time
import os
import numpy as np
import yaml

import torch

from edf_interface.data import PointCloud, SE3, DemoDataset, TargetPoseDemo, preprocess
from edf_interface.data.transforms import quaternion_to_matrix
from diffusion_edf.gnn_data import FeaturedPoints
from diffusion_edf import train_utils
from diffusion_edf.trainer import DiffusionEdfTrainer
from diffusion_edf.visualize import visualize_pose
from diffusion_edf.agent import DiffusionEdfAgent

from scipy.spatial.transform import Rotation as R

class DiffEDFService(Node):
    def __init__(self): 
        super().__init__("diff_edf_service")

        self.device = 'cuda:0'

        first_cb_group = ReentrantCallbackGroup()

        # initialize pick and place agents
        self.init_pick()
        self.init_place()

        self.service = self.create_service(EDF, "diff_edf_service", self.inference_diff_edf, callback_group=first_cb_group)

        self.denoising_configs = dict(
            N_steps_list = [[100, 100], [100, 100, 100]],
            timesteps_list = [[0.04, 0.04], [0.02, 0.02, 0.02]],
            temperatures_list = [[1., 1.], [1., 1., 0.0]],
            log_t_schedule = True,
            diffusion_schedules_list = [
                [[1., 0.15], [0.15, 0.05]],
                [[0.09, 0.03], [0.03, 0.012], [0.012, 0.005]],
            ],
            time_exponent_temp = 1.0,
            time_exponent_alpha = 0.5,
            return_info=True
        )

    def init_pick(self): 
        config_root_dir = 'configs/indy_peg_hole'
        task_type="pick"
        with open(os.path.join(config_root_dir, 'agent.yaml')) as f:
            model_kwargs = yaml.load(f, Loader=yaml.FullLoader)
            model_kwargs_list = model_kwargs['model_kwargs'][f"{task_type}_models_kwargs"]
            try:
                critic_kwargs = model_kwargs['model_kwargs'][f"{task_type}_critic_kwargs"]
            except:
                critic_kwargs = None

        with open(os.path.join(config_root_dir, 'preprocess.yaml')) as f:
            preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
            unprocess_config = preprocess_config['unprocess_config']
            preprocess_config = preprocess_config['preprocess_config']

        agent = DiffusionEdfAgent(
            model_kwargs_list=model_kwargs_list,
            preprocess_config=preprocess_config,
            unprocess_config=unprocess_config,
            device=self.device,
            critic_kwargs=critic_kwargs
        )
        self.pick_agent = agent 
    
    def init_place(self):
        config_root_dir = 'configs/indy_peg_hole'
        task_type="place"
        with open(os.path.join(config_root_dir, 'agent.yaml')) as f:
            model_kwargs = yaml.load(f, Loader=yaml.FullLoader)
            model_kwargs_list = model_kwargs['model_kwargs'][f"{task_type}_models_kwargs"]
            try:
                critic_kwargs = model_kwargs['model_kwargs'][f"{task_type}_critic_kwargs"]
            except:
                critic_kwargs = None

        with open(os.path.join(config_root_dir, 'preprocess.yaml')) as f:
            preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
            unprocess_config = preprocess_config['unprocess_config']
            preprocess_config = preprocess_config['preprocess_config']

        agent = DiffusionEdfAgent(
            model_kwargs_list=model_kwargs_list,
            preprocess_config=preprocess_config,
            unprocess_config=unprocess_config,
            device=self.device,
            critic_kwargs=critic_kwargs
        ) 
        self.place_agent = agent

    def inference_diff_edf(self, request, response):
        task = request.task
        scene_pcd_path = request.scene_pcd_path
        grasp_pcd_path = request.grasp_pcd_path

        if task == "pick":
            agent = self.pick_agent
            N_samples = 20 # reduce number of samples if too slow or short of memory
        elif task == "place":
            agent = self.place_agent
            N_samples = 10 # reduce number of samples if too slow or short of memory
        else:
            raise ValueError(f"'task_type' must be either 'pick' or 'place', but {task} is given.")

        scene_pcd_o3d = o3d.io.read_point_cloud(scene_pcd_path)
        grasp_pcd_o3d = o3d.io.read_point_cloud(grasp_pcd_path)

        scene_pcd = self.pcd2pt(scene_pcd_o3d)
        grasp_pcd = self.pcd2pt(grasp_pcd_o3d)

        T0 = torch.cat([
            torch.tensor([[1., 0., 0.0, 0.]], device=self.device),
            torch.tensor([[0., 0., 0.3]], device=self.device)
        ], dim=-1).repeat(N_samples, 1)
        Ts_init = SE3(poses=T0).to(self.device)

        if task == "pick": 

            Ts_out_raw, scene_proc, grasp_proc, query_weights, query_edf, query_point, info = self.pick_agent.sample(
                scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, Ts_init=Ts_init,
                **self.denoising_configs
            )
            if 'energy' in info.keys():
                Ts_out, energy = Ts_out_raw, info['energy']
                Ts_out = Ts_out[:,2:-2] # Remove outlier energy poses
                Ts_out = Ts_out[:, (quaternion_to_matrix(Ts_out[-1,:,:4])[...,-1,-1] < 0).nonzero().squeeze(), :] # Remove unreachable poses
                if Ts_out.shape[-2] == 0:
                    success = False
                    raise Exception("No reachable pose generated! Try again")
            else:
                Ts_out = Ts_out_raw

            success = True
            result = Ts_out[-1].detach().cpu().numpy()

        elif task =="place":
            Ts_out_raw, scene_proc, grasp_proc, query_weights, query_edf, query_point, info = self.place_agent.sample(
                scene_pcd=scene_pcd, grasp_pcd=grasp_pcd, Ts_init=Ts_init,
                **self.denoising_configs)
        
            if 'energy' in info.keys():
                Ts_out, energy = Ts_out_raw, info['energy']
                Ts_out = Ts_out[:,0:-2] # Remove outlier energy poses #TODO: Check if we are going to use 0:-2 or 2:-2
                Ts_out = Ts_out[:, (quaternion_to_matrix(Ts_out[-1,:,:4])[...,-1,-1] < 0).nonzero().squeeze(), :] # Remove unreachable poses
                if Ts_out.shape[-2] == 0:
                    success = False
                    raise Exception("No reachable pose generated! Try again")
            else:
                Ts_out = Ts_out_raw
            
            success = True
            result = Ts_out[-1].detach().cpu().numpy()

        position = result[4:] * 0.001
        quaternion = result[:4]

        r = R.from_quat(quaternion, scalar_first = True)
        orientation = r.as_euler('xyz', degrees=True)

        response.position = position # value in XYZ, (mm)
        response.orientation = orientation # degree in XYZ (deg)
        response.success = success

    def pcd2pt(self, pcd_o3d): 

        device = self.device
        pcd_points = torch.tensor(np.asarray(pcd_o3d.points)).to(device).to(torch.float32)
        pcd_colors = torch.tensor(np.asarray(pcd_o3d.colors)).to(device).to(torch.float32)

        pcd = PointCloud(pcd_points, pcd_colors)
        return pcd


def main(args=None):
    rclpy.init(args=args)
    diff_edf_service = DiffEDFService()

    try:
        diff_edf_service.get_logger().info('Beginning client, shut down with CTRL-C')
        rclpy.spin(diff_edf_service)
    except KeyboardInterrupt:
        diff_edf_service.get_logger().info('Keyboard interrupt, shutting down.\n')
        
    diff_edf_service.destroy_node()

    rclpy.shutdown()