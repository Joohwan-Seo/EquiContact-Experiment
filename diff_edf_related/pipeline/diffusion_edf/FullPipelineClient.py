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
        self.robot_sub = self.create_subscription(RoboData, "/indy/state", self.robot_state_cb, 10)
        self.robot_pub = self.create_publisher(CompAct, "/indy/act2", 10)
        self.act_pub = self.create_publisher(ActActivity, "act_activity", 10)
        self.request = EDF.Request()
        self.robot_state = {}

    def robot_state_cb(self, msg):
        self.robot_state["ppos"] = np.array([msg.ppos.linear.x, msg.ppos.linear.y, msg.ppos.linear.z, msg.ppos.angular.x, msg.ppos.angular.y, msg.ppos.angular.z])
        self.robot_state["ft"] = np.array([msg.ft.linear.x, msg.ft.linear.y, msg.ft.linear.z, msg.ft.angular.x, msg.ft.angular.y, msg.ft.angular.z])
    
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
    
    def wait_until_reached(self, goal): 
        while np.sum(np.abs(self.robot_state["ppos"][:3] - goal[:3])) > 5: 
            error =  np.sum(self.robot_state["ppos"][:3] - goal[:3])
            self.get_logger().info(f"Waiting for robot to reach goal, error: { error}, goal: {goal}, current: {self.robot_state['ppos']}")
            time.sleep(0.01)

class ACTInferenceClient(Node): 

    def __init__(self): 
        super().__init__("act_inference_client_async")
        self.client = self.create_client(ACTInference, "act_inference")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.request = ACTInference.Request()
        self.robot_pub = self.create_publisher(CompAct, "/indy/act2", 10)
        self.robot_sub = self.create_subscription(RoboData, "/indy/state", self.robot_state_cb, 10)
        self.robot_state = {}

    def robot_state_cb(self, msg):
        self.robot_state["ppos"] = np.array([msg.ppos.linear.x, msg.ppos.linear.y, msg.ppos.linear.z, msg.ppos.angular.x, msg.ppos.angular.y, msg.ppos.angular.z])
        self.robot_state["ft"] = np.array([msg.ft.linear.x, msg.ft.linear.y, msg.ft.linear.z, msg.ft.angular.x, msg.ft.angular.y, msg.ft.angular.z])

    def send_request(self, task, reference_frame): 
        self.request.task = task
        print(f"Reference frame: {list(reference_frame)}")
        self.request.reference_frame = list(reference_frame)
        self.get_logger().info(f"Sending request: {self.request}")
        return self.client.call_async(self.request)


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
    platform = "tilted"

    rclpy.init()
    edf_client = FullPipelineClient()
    indy = IndyDCP3("10.208.112.143")
    act_client = ACTInferenceClient()
    home_pose = np.array([350.0, -186.5, 522.1, 180.0, 0.0, 90.0])
    offset_length = 80.0
    move_to_free(edf_client, home_pose, gripper_state=1)
    time.sleep(5)

    act_activity = ActActivity()
    
    # # #### PICK EXECUTION ####
    # future = edf_client.get_edf("pick")
    # edf_client.get_logger().info("Waiting for pick service")
    # rclpy.spin_until_future_complete(edf_client, future)
    # response = future.result()

    # pick_pos, pick_ori = process_pick_edf_response(response)

    # print(f"Pick pose positions:\n {pick_pos}, and orientations:\n {pick_ori}")

    # # time.sleep(3)

    # R_init = R.from_euler('xyz', pick_ori, degrees=True).as_matrix()
    # offset = R_init @ np.array([0, 0, -offset_length])
    # pick_command = [pick_pos[0] + offset[0],
    #                 pick_pos[1] + offset[1],
    #                 pick_pos[2] + offset[2],
    #                 pick_ori[0],
    #                 pick_ori[1],
    #                 pick_ori[2]]
    
    # edf_client.get_logger().info(f"Pick command: {pick_command}")
    
    # move_to_free(edf_client, pick_command, gripper_state=0)
    # time.sleep(7)

    # pick_pose = np.concatenate((pick_pos, pick_ori)).astype(np.float64).tolist()

    # # os.system(f"python /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/ACTInference_renew.py --reference_frame " + str(pick_pose[0]) 
    # #           + " " + str(pick_pose[1]) + " " + str(pick_pose[2]) + " " + str(pick_pose[3]) + " " + str(pick_pose[4]) + " " + str(pick_pose[5]) +
    # #           " " + "--config /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pick_02.yaml")

    # future = act_client.send_request("pick", pick_pose)
    # rclpy.spin_until_future_complete(act_client, future)
    # response = future.result()
    # time.sleep(1)

    # pose_after_pick = indy.get_control_data()['p']

    # # ### Grasp the peg
    # move_to_free(edf_client, pose_after_pick, gripper_state=1)
    # time.sleep(3)

    # command_1 = np.array([pose_after_pick[0], pose_after_pick[1], pose_after_pick[2] + 150.0, 180.0, 0.0, 90.0], dtype = np.float64)

    # # #### MOVE TO HOME POSE ####
    # move_to_free(edf_client, command_1, gripper_state=1)
    # time.sleep(2)

    # command_2 = np.array([500.0, 0.0, 400, 180.0, 0.0, 90.0], dtype = np.float64)

    # move_to_free(edf_client, command_2, gripper_state=1)
    # time.sleep(2)

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
    time.sleep(5)

    place_pose = np.concatenate((place_pos, place_ori)).astype(np.float64).tolist()
    # os.system(f"python /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/ACTInference_renew.py --reference_frame " + str(place_pose[0]) 
    #           + " " + str(place_pose[1]) + " " + str(place_pose[2]) + " " + str(place_pose[3]) + " " + str(place_pose[4]) + " " + str(place_pose[5]) +
    #           " " + "--config /home/horowitzlab/ros2_ws_clean/src/inference_pipeline/inference_pipeline/configs/ACT_realrobot_pattern_platform_04.yaml")
    future = act_client.send_request("place", place_pose)
    rclpy.spin_until_future_complete(act_client, future)
    response = future.result()

def test_act_client():
    rclpy.init() 
    act_client = ACTInferenceClient()
    # move_to_free(act_client, np.array([457.2, -304.8, 280+80, 180, 0, 90], dtype=np.float64), gripper_state=0)
    # time.sleep(5)
    # future = act_client.send_request("pick", np.array([457.2, -304.8, 280, 180, 0, 90], dtype=np.float64))
    # rclpy.spin_until_future_complete(act_client, future)
    # response = future.result()
    # act_client.get_logger().info(f"Response: {response}")
    # time.sleep(3)
    place_target = np.array([493.21862525, -42.05659921, 309.02251435, 146.1773516,   -4.37193888, 178.6046842 ], dtype=np.float64)

    place_target_before = place_target.copy()

    R_init = R.from_euler('xyz', place_target[3:], degrees=True).as_matrix()
    offset = R_init @ np.array([0, 0, -70])
    place_target_before[:3] += offset

    move_to_free(act_client, place_target_before, gripper_state=1)
    time.sleep(5)
    future = act_client.send_request("place", place_target)
    rclpy.spin_until_future_complete(act_client, future)


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

if __name__ == "__main__": 
    # main("pick")
    # time.sleep(3) 
    # main("place") 
    test_act_client()
    # test_edf_client()
    # full_inference()