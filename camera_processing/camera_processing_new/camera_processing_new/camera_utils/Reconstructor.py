from camera_processing_new.camera_utils.get_pcd_client import PCDClientAsync
import open3d as o3d
import rclpy

from rclpy.node import Node
from argparse import ArgumentParser
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from neuromeka import IndyDCP3, EndtoolState
import copy
import datetime
from scipy.spatial.transform import Rotation as Rot
import time

class Reconstructor: 

    def __init__(self, cam_extrinsics, save_dir): 
        
        self.cam_extrinsics = cam_extrinsics
        self.save_dir = save_dir
        self.gripper_pcds = []
    
    def getse3(self):
        pose = self.indy.get_control_state()['p']
        g = np.eye(4)
        g[:3,:3] = Rot.from_euler('xyz',pose[3:],degrees=True).as_matrix()
        g[:3,3] = np.array(pose[:3]) * 0.001
        return g
    
    def g_rotz(self,theta): 
        # rotation matrix about z axis
        g= np.zeros((4,4))
        g[3,3] = 1 
        g[:3,:3] = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        return g
    
    def merge_scene_pcds(self, pcd1, pcd2 , min_bound = None, max_bound = None):
       

        if min_bound is None and max_bound is None:
            # Default value
            min_bound = np.array([0.15, -0.4, -0.03])
            max_bound = np.array([0.7, 0.55, 0.25])

        
        # Transform the point clouds to spatial frame
        pcd1.transform(self.cam_extrinsics['camera_01'])
        pcd2.transform(self.cam_extrinsics['camera_02'])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # o3d.visualization.draw_geometries([pcd1 + pcd2, mesh_frame])

        bbox_orig = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd1_orig = pcd1.crop(bbox_orig)
        pcd2_orig = pcd2.crop(bbox_orig)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd1 = pcd1.crop(bbox)
        pcd2 = pcd2.crop(bbox)

        # Run ICP
        threshold = 0.001

        # Visualize pcd 1 and pcd2 separately
        # o3d.visualization.draw_geometries([pcd1, mesh_frame])
        # o3d.visualization.draw_geometries([pcd2, mesh_frame])

        initial_transformation = np.eye(4)
        initial_transformation[:3, 3] = np.array([0.005, -0.005, 0.0])

        rotmat_x = R.from_euler('x', -1, degrees=True).as_matrix()

        rotmat_z = R.from_euler('z', -1, degrees=True).as_matrix()
        initial_transformation[:3, :3] = rotmat_z @ rotmat_x

        reg_p2l = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        )

        print(reg_p2l.transformation)

        # Transform and merge
        pcd2.transform(reg_p2l.transformation)
        pcd1 += pcd2

        # pcd2_orig.transform(reg_p2l.transformation)
        pcd1_orig += pcd2_orig
        
        # Remove outlier
        cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd1 = pcd1.select_by_index(ind)

        cl, ind = pcd1_orig.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd1_orig = pcd1_orig.select_by_index(ind)

        # Voxel downsampling
        voxel_size = 0.001
        pcd1 = pcd1.voxel_down_sample(voxel_size)

        # Add the coordinate frame
        # o3d.visualization.draw_geometries([pcd1, mesh_frame])
        o3d.visualization.draw_geometries([pcd1_orig, mesh_frame])

        # Save the merged point cloud
        
        return pcd1_orig
    
    def merge_gripper_pcds(self, gripper_pcds, eef_poses, angles, save = False, crop_lims = (np.array([[0.1, -0.8, 0.17],[0.4, 0, 0.9]])), voxel_size = 0.001): 
        
        # Transform gripper pcds
        pcds_comb = []
        bbox = o3d.geometry.AxisAlignedBoundingBox(crop_lims[0], crop_lims[1])

        # Transform gripper pcds to eef frame
        for i in range(len(gripper_pcds)): 
            eef_pcd = gripper_pcds[i]
            eef_pcd = eef_pcd.crop(bbox)

            # eef_pcd.translate(np.array([-0.015, -0.025, 0.0]))

            # Visualize each gripper pcd
            meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0.2])
            # o3d.visualization.draw_geometries([eef_pcd, meshframe], f"Gripper PCD {i} before rotation")

            eef_pcd.transform(np.linalg.inv(eef_poses[i]))

            # offset = np.array([0.014, 0.022, 0.0])
            offset = np.array([0.001, -0.001, 0.0])
            rotmat_z = R.from_euler('z', -angles[i], degrees=False).as_matrix()
            offset_rotated = (rotmat_z @ offset.reshape((-1,1))).reshape(-1)

            eef_pcd.translate(offset_rotated)

            # o3d.visualization.draw_geometries([eef_pcd, meshframe], f"Gripper PCD {i} after rotation")

            # processed_pcd = processed_pcd.transform(self.g_rotz(-angles[i]))
            
            pcds_comb.append(eef_pcd)

        meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Merge gripper pcds
        gripper_pcd_comb = pcds_comb[0] + pcds_comb[1] + pcds_comb[2] + pcds_comb[3]

        # Remove outlier
        cl, ind = gripper_pcd_comb.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        gripper_pcd_comb = gripper_pcd_comb.select_by_index(ind)

        # Visualize
        o3d.visualization.draw_geometries([gripper_pcd_comb, meshframe], "Gripper PCD Combined")

        # Save the merged gripper pcd
        gripper_pcd_path = self.save_path 
        if f'demo{self.demo_num}' not in self.demos.keys():
            self.demos[f'demo{self.demo_num}'] = {}
        self.demos[f'demo{self.demo_num}']['gripper'] = gripper_pcd_path
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "gripper_pcd.pcd"), gripper_pcd_comb)

        # Reset gripper pcds
        self.gripper_pcds = []
        return gripper_pcd_path