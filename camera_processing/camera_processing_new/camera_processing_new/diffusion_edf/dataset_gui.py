from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField
from neuromeka import IndyDCP3
from neuromeka import JointTeleopType, TaskTeleopType
from neuromeka import BlendingType, EndtoolState
from camera_processing_new.camera_utils.Reconstructor import Reconstructor
import torch 
import customtkinter
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import os
import time
NO_BLENDING = 0
OVERRIDE_BLENDING = 1
DUPLICATE_BLENDING = 2
ABSOLUTE_JOINT = 0
RELATIVE_JOINT = 1


class GatherData(Node): 

    def __init__(self, home_pos, scene_pos, gripper_pos, voxel_size = 0.005): 
        super.__init__(self, "gather_data")
        self.home_pos = home_pos 
        self.scene_pos = scene_pos
        self.gripper_pos = gripper_pos 
        self.demo = 0
        self.wd = "/home/horowitzlab/ros2_ws/src/camera_processing/camera_processing/diffusion_edf/new_data"
        self.grasp_dir = os.path.join(self.wd, "grasp_poses")
        self.grasp_pcd = os.path.join(self.wd, "grasp_pcd")
        self.place_dir = os.path.join(self.wd, "place_poses")
        self.place_pcd = os.path.join(self.wd, "place_pcd")

        self.ip_addr = "10.208.112.143"
        self.indy = IndyDCP3(self.ip_addr)
        extrinsics = {"camera_01": np.load("/home/horowitzlab/ros2_ws_clean/test_pcd/cad_icp_camera_01_square_v2.npy"),
                      "camera_02": np.load("/home/horowitzlab/ros2_ws_clean/test_pcd/cad_icp_camera_02_square_v3.npy")}
        self.reconstructor = Reconstructor(extrinsics, "/home/horowitzlab/ros2_ws_clean/test_pcd/scene_pcd_trial.pcd")
        #### Subscribers #### 
        self.pcd_subs = {}
        self.pcd_raw = {}
        self.pcd_subs['camera_01'] = self.create_subscription(PointCloud2, "camera_01/depth_registered/points", partial(self.pcd_cb, camera="camera_01"), 10)
        self.pcd_subs['camera_02'] = self.create_subscription(PointCloud2, "camera_02/depth_registered/points", partial(self.pcd_cb, camera="camera_02"), 10)

        self.voxel_size = voxel_size

    def pcd_cb(self, msg, camera):
        self.pcd_raw[camera] = msg 
    
    def pcd2pt(self, pcd_o3d): 

        device = self.device
        pcd_points = torch.tensor(np.asarray(pcd_o3d.points)).to(device).to(torch.float32)
        pcd_colors = torch.tensor(np.asarray(pcd_o3d.colors)).to(device).to(torch.float32)
        return [pcd_points, pcd_colors]
    
    def get_scene_pcd(self): 
        pcd1_msg = self.pcd_raw['camera_01']
        pcd2_msg = self.pcd_raw['camera_02']
        
        pcd1 = self.pcd_converter.pcdmsg2pcd(pcd1_msg)
        pcd2 = self.pcd_converter.pcdmsg2pcd(pcd2_msg)
        scene_pcd = self.reconstructor.merge_scene_pcds(pcd1, pcd2)
        return scene_pcd
    
    def robot_reset(self): 
        self.indy.movej(self.home_pos)
        self.demo+=1

    def get_scene(self): 
        self.indy.movej(self.scene_pos)
        self.reconstructor.merge_scene_pcds()
    
    def get_gripper(self): 
        self.indy.movej(self.gripper_pos)
        self.reconstructor.use_camera_1_only = True 
        self.reconstructor.apply_ICP = False
        reconstruct_gripper(self.reconstructor, voxel_size = self.voxel_size)

    def open_gripper(self):
        signal = [{'port': 'C', 'states': [EndtoolState.LOW_PNP]}]
        self.indy.set_endtool_do(signal)

    def close_gripper(self):
        signal = [{'port': 'C', 'states': [EndtoolState.HIGH_PNP]}]
        self.indy.set_endtool_do(signal)

    def record_grasp(self): 
        eef_pose = np.array(self.indy.get_control_data()['p'])
        print(os.path.join(self.grasp_dir, "demo_" + str(self.demo) + ".npy"))
        np.save(os.path.join(self.grasp_dir, "demo_" + str(self.demo) + ".npy"), eef_pose)

    def record_place(self): 
        eef_pose = np.array(self.indy.get_control_data()['p'])
        np.save(os.path.join(self.place_dir, "demo_" + str(self.demo)+".npy"), eef_pose)

    def start(self):
        customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        app = customtkinter.CTk()  # create CTk window like you do with the Tk window
        app.geometry("400x240")
        button2 = customtkinter.CTkButton(master=app, text="Reset", command=self.robot_reset, width=5, height=5)
        button2.place(relx=0.5, rely=0.1, anchor=customtkinter.CENTER)
        button = customtkinter.CTkButton(master=app, text="Get Scene", command=self.get_scene, width=5, height = 5)
        button.place(relx=0.5, rely=0.25, anchor=customtkinter.CENTER)
        button4 = customtkinter.CTkButton(master=app, text="Get Grasp", command=self.record_grasp, width=5, height=5)
        button4.place(relx=0.5, rely=0.4, anchor=customtkinter.CENTER)
        button3 = customtkinter.CTkButton(master=app, text="Get Gripper", command=self.get_gripper, width=5, height=5)
        button3.place(relx=0.5, rely=0.55, anchor=customtkinter.CENTER)
        button5 = customtkinter.CTkButton(master=app, text="Get Place", command=self.record_place, width=5, height=5)
        button5.place(relx=0.5, rely=0.7, anchor=customtkinter.CENTER)

        button6 = customtkinter.CTkButton(master=app, text="Open Gripper", command=self.open_gripper, width=5, height=5)
        button6.place(relx=0.3, rely=0.85, anchor=customtkinter.CENTER)

        button7 = customtkinter.CTkButton(master=app, text="Close Gripper", command=self.close_gripper, width=5, height=5)
        button7.place(relx=0.7, rely=0.85, anchor=customtkinter.CENTER)
        
        
        app.mainloop()
        #destroy_pipelines(self.reconstructor.pipelines)
    
if __name__=="__main__": 
    home_pos = [0, 0, -90, 0, -90, 90]
    scene_pos = [0, -20, -70, 0, -60, 90]
    gripper_pos = [0, -20, -70, 0, -90, 90]
    voxel_size = 0.001
    gatherer = GatherData(home_pos, scene_pos, gripper_pos, voxel_size = voxel_size)
    gatherer.start()
    #destroy_pipelines(gatherer.reconstructor.pipelines)    