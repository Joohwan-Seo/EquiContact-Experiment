import rclpy 
from rclpy.node import Node
from camera_interfaces.srv import EDF 
import numpy as np
import time
import os
import open3d as o3d 
import camera_interfaces.srv
from camera_interfaces.msg import CompAct, ActActivity, RoboData
from camera_interfaces.srv import ACTInference
from std_msgs.msg import Bool
from neuromeka import IndyDCP3, EndtoolState
from scipy.spatial.transform import Rotation as R

class FullPipelineClient(Node): 
    def __init__(self): 
        super().__init__("full_pipeline_client")
        self.client = self.create_client(EDF, "diff_edf_service")
        self.robot_pub = self.create_publisher(CompAct, "/indy/act2", 10)
        self.act_pub = self.create_publisher(ActActivity, "act_activity", 10)
        self.request = EDF.Request()
        self.robot_state = {}
    
    def get_gripper_pcd(self, task_type):
        if task_type == "pick":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        elif task_type == "place":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/gripper_pcd_demo4.pcd"
        return gripper_pcd_path

    def get_edf(self, task): 
        self.request = EDF.Request()
        self.request.task = task 
        self.request.scene_pcd_path = "None"

        if task == "pick":
            self.request.grasp_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        elif task == "place":
            self.request.grasp_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/gripper_pcd_demo4.pcd"

        return self.client.call_async(self.request)

class ACTInferenceClient(Node): 
    def __init__(self): 
        super().__init__("act_inference_client_async")
        self.client = self.create_client(ACTInference, "act_inference")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.request = ACTInference.Request()
        self.robot_pub = self.create_publisher(CompAct, "/indy/act2", 10)
        self.robot_state = {}

    def send_request(self, pick_reference_frame, place_reference_frame): 
        self.request.pick_reference_frame = pick_reference_frame
        self.request.place_reference_frame = place_reference_frame
        print(f"Pick Reference frame: {list(pick_reference_frame)}")
        print(f"Place Reference frame: {list(place_reference_frame)}")
        self.get_logger().info(f"Sending request: {self.request}")

        return self.client.call_async(self.request)
    
def process_pick_edf_response(response):
    pick_pos = np.array(response.position).reshape((-1, 3))
    pick_ori = np.array(response.orientation).reshape((-1, 3))

    pick_rotm = R.from_euler('xyz', pick_ori, degrees = True).as_matrix()

    tip_pos = np.zeros((pick_pos.shape[0], 3))
    for i in range(pick_pos.shape[0]):
        tip_pos[i] = pick_pos[i] + pick_rotm[i] @ np.array([0, 0, 190])

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
    elif platform == 'tilted2':
        rotm_default = R.from_euler('xyz', [150, 0, 90], degrees = True).as_matrix()

    tip_pos = np.zeros((place_pos.shape[0], 3))
    for i in range(place_pos.shape[0]):
        tip_pos[i] = place_pos[i] + place_rotm[i] @ np.array([0, 0, 285])
    
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
        # for i in range(4):
        #     dist = np.trace(np.eye(3) - tip_ori_list[i].T @ rotm_default)
        #     if dist < min_dist:
        #         min_dist = dist
        #         tip_ori_mean = tip_ori_list[i]
        if platform == 'tilted':
            psi_des = 0
        elif platform == 'tilted2':
            psi_des = - np.pi / 2
        psi_curr, theta, phi = R.from_matrix(tip_ori_mean).as_euler('zyx', degrees=False)
        tip_ori_mean = R.from_euler('zyx', [psi_des, theta, phi], degrees=False).as_matrix()

    place_pos_processed = tip_pos_mean + tip_ori_mean @ np.array([0, 0, -285])

    place_ori_processed = R.from_matrix(tip_ori_mean).as_euler('xyz', degrees = True)

    return place_pos_processed, place_ori_processed

def full_inference(platform):
    assert platform in ['default', 'tilted', 'tilted2'], "Platform must be either 'default' or 'tilted'"

    edf_client = FullPipelineClient()
    act_client = ACTInferenceClient()

    # #### Get Pick & Place Pose from EDF Client ####
    ### Pick ####
    future = edf_client.get_edf("pick")
    edf_client.get_logger().info("Waiting for pick service")
    rclpy.spin_until_future_complete(edf_client, future)
    response = future.result()

    pick_pos, pick_ori = process_pick_edf_response(response)

    print(f"Pick pose positions:\n {pick_pos}, and orientations:\n {pick_ori}")

    pick_pose = np.concatenate((pick_pos, pick_ori)).astype(np.float64).tolist()

    #### Place ####
    future = edf_client.get_edf("place")
    edf_client.get_logger().info("Waiting for place service")
    rclpy.spin_until_future_complete(edf_client, future)
    response = future.result()

    place_pos, place_ori = process_place_edf_response(response, platform)
    print(f"Place pose positions:\n {place_pos}, and orientations:\n {place_ori}")

    place_pose = np.concatenate((place_pos, place_ori)).astype(np.float64).tolist()

    ### Send pick and place poses to Pipeline+ACT Client ###
    future = act_client.send_request(pick_pose, place_pose)
    act_client.get_logger().info("Waiting for ACT inference service")
    rclpy.spin_until_future_complete(act_client, future)
    response = future.result()

    return


def test_act_client():
    act_client = ACTInferenceClient()

    pick_pose = np.array([457.2, -304.8, 280, 180, 0, 90]).astype(np.float64).tolist()
    place_pose = np.array([478.9906091239746, -17.272042110616994, 322.7776693213716, 151.5666801770288, -2.080297903095018, 175.7491233312564]).astype(np.float64).tolist()
    future = act_client.send_request(pick_pose, place_pose)
    rclpy.spin_until_future_complete(act_client, future)
    response = future.result()

    return


def test_edf_client():
    edf_client = FullPipelineClient()
    # future = edf_client.get_edf("pick")
    # edf_client.get_logger().info("Waiting for pick service")
    # rclpy.spin_until_future_complete(edf_client, future)
    # response = future.result()
    
    # pick_pos, pick_ori = process_pick_edf_response(response)

    # print(f"Pick pose positions:\n {pick_pos}, and orientations:\n {pick_ori}")

    # time.sleep(3)

    platform = 'tilted2'

    future = edf_client.get_edf("place")
    edf_client.get_logger().info("Waiting for place service")
    rclpy.spin_until_future_complete(edf_client, future)
    response = future.result()

    place_pos, place_ori = process_place_edf_response(response, platform)
    print(f"Place pose positions:\n {place_pos}, and orientations:\n {place_ori}")

if __name__ == "__main__": 
    rclpy.init()

    platform = "tilted2"

    # test_act_client()
    full_inference(platform)

    rclpy.shutdown()
    exit(0)
