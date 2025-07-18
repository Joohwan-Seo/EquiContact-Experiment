from camera_processing_new.camera_utils.get_pcd_client import PCDClientAsync
import open3d as o3d
import rclpy
import pynput
from pynput.keyboard import Key, Listener, KeyCode
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


### get_scene_pcds
###     get scene pcd from camera_01 and camera_02
###     merge scene pcd from camera_01 and camera_02
###     save scene pcd
### get grasp pose (indy7)
### get gripper pcd
###     refer to gripper code
### get place pose (indy7)

class DataRecorder(Node):
    def __init__(self, save_dir, num_pcds = 1, delay = 0.2):
        
        super().__init__("data_recorder")
        self.client = PCDClientAsync()
        
        # self.camera_id = camera_id
        self.save_dir = save_dir
        self.num_pcds = num_pcds
        self.delay = delay

        self.scene_pcd = None
        self.scene_pcd_path = None
        self.grasp_pose = None
        self.gripper_pcds = []
        self.gripper_pcds_comb = []
        self.eef_poses = []
        self.place_pose = None
        self.camera_01_scene_pcd_path = None
        self.camera_02_scene_pcd_path = None
        self.gripper_pcd_path = None
        self.demos = {}
        self.demo_num = 0
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        
        self.g1 = np.load("/home/horowitzlab/ros2_ws_clean/test_pcd/cad_icp_camera_01_square_v2.npy")
        self.g2 = np.load("/home/horowitzlab/ros2_ws_clean/test_pcd/cad_icp_camera_02_square_v3.npy")

        # logger to say it is intialized
        self.get_logger().info("Data Recorder Initialized")

        self.ip = "10.208.112.143"
        self.indy = IndyDCP3(self.ip)


    def get_scene_pcds(self):
        # Get scene pcds from camera_01
        # self.joint_move([90, 0, -90, 0, -90, 90])
        future = self.client.send_request("/camera_01", self.save_dir, self.num_pcds, self.delay)
        rclpy.spin_until_future_complete(self.client, future)
        response = future.result()
        if response.success:
            self.get_logger().info(f"save_dir: {self.save_dir}")
            self.get_logger().info("Getting scene pcds from /camera_01 succeeded")
        filename =  f"pcd0_cam1.pcd" 
        self.camera_01_scene_pcd_path  = os.path.join(self.save_dir,filename)
        
        # Get scene pcds from camera_02
        future2 = self.client.send_request("/camera_02", self.save_dir, self.num_pcds, self.delay)
        rclpy.spin_until_future_complete(self.client, future2)
        response2 = future2.result()
        if response2.success:
            self.get_logger().info(f"save_dir: {self.save_dir}")
            self.get_logger().info("Getting scene pcds from /camera_02 succeeded")
        filename =  f"pcd0_cam2.pcd" 
        self.camera_02_scene_pcd_path  = os.path.join(self.save_dir,filename)
        

    def merge_scene_pcds(self, min_bound = None, max_bound = None):
        pcd1 = o3d.io.read_point_cloud(self.camera_01_scene_pcd_path)
        pcd2 = o3d.io.read_point_cloud(self.camera_02_scene_pcd_path)

        if min_bound is None and max_bound is None:
            # default value
            min_bound = np.array([0.15, -0.4, -0.03])
            max_bound = np.array([0.7, 0.55, 0.25])

        
        # transform the point clouds to spatial frame
        pcd1.transform(self.g1)
        pcd2.transform(self.g2)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # o3d.visualization.draw_geometries([pcd1 + pcd2, mesh_frame])

        bbox_orig = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd1_orig = pcd1.crop(bbox_orig)
        pcd2_orig = pcd2.crop(bbox_orig)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd1 = pcd1.crop(bbox)
        pcd2 = pcd2.crop(bbox)

        # run ICP
        threshold = 0.001

        # visualize pcd 1 and pcd2 separately
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

        # transform and merge
        pcd2.transform(reg_p2l.transformation)
        pcd1 += pcd2

        # pcd2_orig.transform(reg_p2l.transformation)
        pcd1_orig += pcd2_orig
        
        # remove outlier
        cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd1 = pcd1.select_by_index(ind)

        cl, ind = pcd1_orig.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd1_orig = pcd1_orig.select_by_index(ind)

        # voxel downsampling
        voxel_size = 0.001
        pcd1 = pcd1.voxel_down_sample(voxel_size)

        # add the coordinate frame
        # o3d.visualization.draw_geometries([pcd1, mesh_frame])
        o3d.visualization.draw_geometries([pcd1_orig, mesh_frame])

        # save the merged point cloud
        scene_pcd_path = os.path.join(self.save_dir, f"scene_pcd_demo{self.demo_num}.pcd")
        o3d.io.write_point_cloud(scene_pcd_path, pcd1)
        if f'demo{self.demo_num}' not in self.demos.keys():
            self.demos[f'demo{self.demo_num}'] = {}
        self.demos[f'demo{self.demo_num}']['scene'] = scene_pcd_path
        self.get_logger().info(f"Scene pcd saved to {os.path.join(self.save_dir, f'scene_pcd.pcd')}")
        return scene_pcd_path

    def get_pose(self, type):
        pose = self.indy.get_control_state()['p']
        if f'demo{self.demo_num}' not in self.demos.keys():
            self.demos[f'demo{self.demo_num}'] = {}
        self.demos[f'demo{self.demo_num}'][type] = pose
        if type == "grasp": 
            self.get_logger().info(f" Current Grasp pose: {pose}")
        elif type == "place": 
            self.get_logger().info(f" Current Place pose: {pose}")
            
    def collect_gripper_pcd(self): 
        # 4 joint targets
        j_target_init = [-10,-8, -90, -10, -80, 195]
        j_targets = [[-10,-8, -90, -10, -80, 195], [-10,-8, -90, -10, -80, 105], [-10,-8, -90, -10, -80, 15], [-10,-8, -90, -10, -80, -75]]
        
        # 4 angles for joint 6
        angles_list = []
        for i in range(len(j_targets)):
            angle6 = (j_targets[i][-1] - 195) / 180 * np.pi
            angles_list.append(angle6)

        angles_list = np.array([0, -90, -180, -270]) /180 * np.pi
        
        self.get_logger().info(f"Angles list: {angles_list}")
        self.get_gripper_pcd(j_target_init, j_targets, angles_list)
    
    def get_gripper_pcd(self, j_target_init = [-10,-8, -90, -10, -80, 195],
                        j_targets = [[-10,-8, -90, -10, -80, 195], [-10,-8, -90, -10, -80, 105], [-10,-8, -90, -10, -80, 15], [-10,-8, -90, -10, -80, -75]], 
                        angles = np.array([0, -90, -180, -270]) /180 * np.pi, 
                        save=False, 
                        crop_lims=(np.array([[0.2, -0.5, -0.05],[0.8, 0.6, 0.5]])), voxel_size = 0.001): 
        # Move to target pose
        self.indy.movej(j_target_init)
        print("Getting Gripper Point Cloud")

        # Get gripper pcds for each joint target
        for i in range(len(j_targets)):
            self.joint_move(j_targets[i])
            time.sleep(1)
            pcds = self.get_gripper_angles(angles[i])

        # Merge gripper pcds
        gripper_pcd_path = self.merge_gripper_pcds(j_target_init, angles, save)
        return gripper_pcd_path

    def joint_move(self, jtarget):
        # Move to joint target, wait until target is reached
        flag = 0
        while(True):
            if flag == 0:
                self.indy.movej(jtarget = jtarget)
                while(True):
                    if self.indy.get_motion_data()['is_target_reached'] == False:
                        flag = 1
                        break
            else:
                if self.indy.get_motion_data()['is_target_reached']:
                    flag = 0
                    break

    def get_gripper_angles(self, angle):
        # Get gripper pcds
        future = self.client.send_request("/camera_01", self.save_dir, self.num_pcds, self.delay)
        rclpy.spin_until_future_complete(self.client, future)
        response = future.result()
        if response.success:
            self.get_logger().info(f"save_dir: {self.save_dir}")
            self.get_logger().info("Getting gripper full pcd from /camera_01 succeeded")
        
        # Append gripper pcds
        pcd = o3d.io.read_point_cloud(os.path.join(self.save_dir, f"pcd0_cam1.pcd"))
        pcd = pcd.transform(self.g1)

        self.eef_poses.append(self.getse3())
        self.gripper_pcds.append(pcd)

        self.get_logger().info(f"Got gripper pcd for angle {angle}")
        # o3d.visualization.draw_geometries([o3d.io.read_point_cloud(os.path.join(self.save_dir, f"pcd0_cam1.pcd"))])

    def getse3(self):
        pose = self.indy.get_control_state()['p']

        g = np.eye(4)
        g[:3,:3] = Rot.from_euler('xyz',pose[3:],degrees=True).as_matrix()
        g[:3,3] = np.array(pose[:3]) * 0.001
        return g

    def merge_gripper_pcds(self, j_target_init, angles, save = False, crop_lims = (np.array([[0.1, -0.8, 0.17],[0.4, 0, 0.9]])), voxel_size = 0.001): 
        
        # transform gripper pcds
        pcds_comb = []
        bbox = o3d.geometry.AxisAlignedBoundingBox(crop_lims[0], crop_lims[1])

        # transform gripper pcds to eef frame
        for i in range(len(self.gripper_pcds)): 
            eef_pcd = self.gripper_pcds[i]
            eef_pcd = eef_pcd.crop(bbox)

            # eef_pcd.translate(np.array([-0.015, -0.025, 0.0]))

            # visualize each gripper pcd
            meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0.2])
            # o3d.visualization.draw_geometries([eef_pcd, meshframe], f"Gripper PCD {i} before rotation")

            eef_pcd.transform(np.linalg.inv(self.eef_poses[i]))

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

        # remove outlier
        cl, ind = gripper_pcd_comb.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        gripper_pcd_comb = gripper_pcd_comb.select_by_index(ind)

        # visualize
        o3d.visualization.draw_geometries([gripper_pcd_comb, meshframe], "Gripper PCD Combined")

        # save the merged gripper pcd
        gripper_pcd_path = os.path.join(self.save_dir, f"gripper_pcd_demo{self.demo_num}.pcd")
        if f'demo{self.demo_num}' not in self.demos.keys():
            self.demos[f'demo{self.demo_num}'] = {}
        self.demos[f'demo{self.demo_num}']['gripper'] = gripper_pcd_path
        o3d.io.write_point_cloud(os.path.join(self.save_dir, gripper_pcd_path), gripper_pcd_comb)

        # reset gripper pcds
        self.gripper_pcds = []
        return gripper_pcd_path
    
    def g_rotz(self,theta): 
        # rotation matrix about z axis
        g= np.zeros((4,4))
        g[3,3] = 1 
        g[:3,:3] = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        return g

    def on_press(self, key):
        if key == KeyCode(char='1'):
            self.get_logger().info("Getting Scene Point Clouds") 
            self.get_scene_pcds()
            self.merge_scene_pcds()

        elif key == KeyCode(char='2'): 
            self.get_logger().info("Getting Grasp Pose")
            self.get_pose("grasp")

        elif key == KeyCode(char='3'):
            self.get_logger().info("Collecting Gripper PCD")
            self.collect_gripper_pcd()

        elif key == KeyCode(char='4'): 
            self.get_logger().info("Getting Place Pose")
            self.get_pose("place")

        elif key == KeyCode(char='5'):
            self.get_logger().info(f"Saving data, current demo number:{self.demo_num}")
            self.demo_num += 1


        elif key == KeyCode(char='0'):
            self.get_logger().info("Opening Gripper")
            signal = [{'port': 'C', 'states': [EndtoolState.LOW_PNP]}]
            self.indy.set_endtool_do(signal)
        elif key == KeyCode(char='9'):
            self.get_logger().info("Closing Gripper")
            signal = [{'port': 'C', 'states': [EndtoolState.HIGH_PNP]}]
            self.indy.set_endtool_do(signal)
        elif key == KeyCode(char='h'):
            self.get_logger().info("Moving to Home Position")
            self.indy.movej([0, 0, -90, 0, -90, 90])

    def on_release(self,key):
        if key == Key.esc:
            self.get_logger().info("ESC pressed, ending demo")
            print(self.demos)
            with open(os.path.join(self.save_dir, f"demo_data{datetime.datetime.now()}.pkl"), 'wb') as f:
                pickle.dump(self.demos, f)
            # Stop listener
            return False

def main():
    rclpy.init()
    parser = ArgumentParser()
    parser.add_argument("--save_dir", default = "/home/horowitzlab/ros2_ws_clean/data/pcds")
    parser.add_argument("--num_pcds", default=1)
    parser.add_argument("--delay", default = 0.2)
    args = parser.parse_args()
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir, exist_ok=True)
        
    recorder = DataRecorder(save_dir = args.save_dir, num_pcds = int(args.num_pcds), delay = float(args.delay))
    with recorder.listener as listener:
        listener.join()
    return

if __name__ == "__main__":
    main()