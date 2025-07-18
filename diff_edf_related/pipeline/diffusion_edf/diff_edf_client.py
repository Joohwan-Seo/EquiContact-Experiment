import rclpy 
from rclpy.node import Node
from camera_interfaces.srv import EDF 
from camera_processing_new.camera_utils.get_edf_dataset import DataRecorder
import numpy as np

import open3d as o3d 

from neuromeka import IndyDCP3, EndtoolState

class DiffEDFClient(Node): 
    
    def __init__(self): 
        super().__init__("diff_edf_client")
        self.datarecorder = DataRecorder("/home/horowitzlab/ros2_ws/test_pcds3")
        self.client = self.create_client(EDF, "diff_edf_service")
        self.request = EDF.Request()

    def get_scene_pcd(self): 
        self.datarecorder.get_scene_pcds()
        min_bound = np.array([0.3, -0.25, -0.03])
        max_bound = np.array([0.7, 0.4, 0.3])
        merged_scene_path = self.datarecorder.merge_scene_pcds(min_bound, max_bound)

        return merged_scene_path
    
    def get_gripper_pcd(self, task_type):
        if task_type == "pick":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        elif task_type == "place":
            gripper_pcd_path = self.datarecorder.get_gripper_pcd()
        return gripper_pcd_path

    def get_edf(self, task): 
        scene_pcd_path = self.get_scene_pcd() 
        gripper_pcd_path = self.get_gripper_pcd(task)

        self.request = EDF.Request()
        self.request.task = task 
        self.request.scene_pcd_path = scene_pcd_path
        self.request.grasp_pcd_path = gripper_pcd_path

        return self.client.call_async(self.request)
    
def main(task):
    rclpy.init()
    client = DiffEDFClient()
    future = client.get_edf(task)
    rclpy.spin_until_future_complete(client, future)
    response = future.result()
    client.get_logger().info(f"{task} position: {response.position}, orientation:{response.orientation}")

if __name__ == "__main__": 
    main("pick") 
    # main("place") 