import rclpy 
from rclpy.node import Node
from camera_interfaces.srv import EDF 
import numpy as np

import open3d as o3d 
from camera_interfaces.msg import CompACT
from std_msgs.msg import Bool
from neuromeka import IndyDCP3, EndtoolState

class DiffEDFClient(Node): 
    
    def __init__(self): 
        super().__init__("diff_edf_client")
        self.client = self.create_client(EDF, "diff_edf_service")
        self.robot_pub = self.create_publisher(CompACT, "/indy/act2", 10)
        self.act_pub = self.create_publisher(Bool, "act_activity", 10)
        self.request = EDF.Request()

    def get_scene_pcd(self): 
        self.datarecorder.get_scene_pcds()
        min_bound = np.array([0.3, -0.2, -0.03])
        max_bound = np.array([0.7, 0.2, 0.3])
        merged_scene_path = self.datarecorder.merge_scene_pcds(min_bound, max_bound)

        return merged_scene_path
    
    def get_gripper_pcd(self, task_type):
        if task_type == "pick":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        elif task_type == "place":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/gripper_pcd_demo5.pcd"
        return gripper_pcd_path

    def get_edf(self, task): 
        self.request = EDF.Request()
        self.request.task = task 
        self.request.scene_pcd_path = "None"
        self.request.grasp_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"

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