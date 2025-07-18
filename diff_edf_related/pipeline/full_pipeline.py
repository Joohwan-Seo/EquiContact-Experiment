import rclpy 
from rclpy.node import Node
from camera_interfaces.srv import EDF 
from camera_processing_new.camera_utils.get_edf_dataset import DataRecorder

from pipeline.gain_scheduling_policy.BC_policy import BCInference

import open3d as o3d 

from neuromeka import IndyDCP3, EndtoolState

import time
import numpy as np

from scipy.spatial.transform import Rotation as Rot


class FullPipeline(Node): 
    
    def __init__(self): 
        super().__init__("diff_edf_client")
        self.datarecorder = DataRecorder("/home/horowitzlab/ros2_ws/test_pcds3")
        self.diff_edf_client = self.create_client(EDF, "diff_edf_service")
        self.request = EDF.Request()

        self.ip = "10.208.112.127"
        self.indy = IndyDCP3(self.ip)

        #self.BC_policy = BCInference()

        self.get_logger().info("Activating SDK")

        self.offset = np.array([5, 5, 5, 0, 0, 0]) # in mm and degrees


        mode = 1
        self.indy.set_custom_control_mode(mode=mode)
        mode_result = self.indy.get_custom_control_mode()

        if mode_result["mode"] == "1":
            self.indy.set_custom_control_mode(0)
            self.get_logger().info("Custom control mode is set to 0")
        else:
            try:
                self.indy.activate_sdk(license_key = 'C525EDD4542A725BCD37755E648D46CCE9694A72852EB05390C7D150F99966DA', expire_date = '2025-12-31')
            except Exception as e:
                self.get_logger().error(f"Failed to activate sdk: {e}")

        self.scene_pcd_path = None

        self.FT_bias = self.get_ft_bias()

    def get_ft_bias(self):
        # take FT sensor value for 5 seconds and make an average

        FT_list = []        

        tic = time.time()
        while time.time() - tic < 5:
            FT_sensor_dict = self.indy.get_ft_sensor_data()
            FT_sensor_list = [FT_sensor_dict['ft_Fx'], FT_sensor_dict['ft_Fy'], FT_sensor_dict['ft_Fz'], 
                              FT_sensor_dict['ft_Tx'], FT_sensor_dict['ft_Ty'], FT_sensor_dict['ft_Tz']]
            FT_list.append(FT_sensor_list)

        FT_list = np.array(FT_list)
        
        FT_bias = np.mean(FT_list, axis = 0)

        return FT_bias        

    def get_scene_pcd(self): 
        self.datarecorder.get_scene_pcds()
        min_bound = np.array([0.3, -0.2, -0.03])
        max_bound = np.array([0.7, 0.2, 0.3])
        merged_scene_path = self.datarecorder.merge_scene_pcds(min_bound = min_bound, max_bound = max_bound)

        return merged_scene_path
    
    def get_gripper_pcd(self, task_type):
        if task_type == "pick":
            gripper_pcd_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        elif task_type == "place":
            gripper_pcd_path = self.datarecorder.get_gripper_pcd()
        return gripper_pcd_path

    def get_edf(self, task):
        assert task in ["pick", "place"], "Invalid task type"

        if task == "pick" or self.scene_pcd_path is None:
            self.scene_pcd_path = self.get_scene_pcd()

        self.gripper_pcd_path = self.get_gripper_pcd(task) # just for unified naming convention

        self.request = EDF.Request()
        self.request.task = task 
        self.request.scene_pcd_path = self.scene_pcd_path
        self.request.grasp_pcd_path = self.gripper_pcd_path

        return self.diff_edf_client.call_async(self.request)
    
    def exec_pick(self, target_pose):
        #TODO(JS) Implement this
        self.gripper_command("open")
        # Move to home position
        self.indy.movej([0, 0, -90, 0, -90, 90])
        target_pose = target_pose + self.offset
        # Move above the target pose and then move to the target pose
        self.move_robot(target_pose + np.array([0, 0, 50, 0, 0, 0]))
        time.sleep(3)
        # self.move_robot(target_pose + np.array([-15, 0, 0, 0, 0, 0]))
        self.move_robot(target_pose)

        pass
    
    def exec_place(self, target_pose):
        # Move to home position
        
        self.indy.movej([0, 0, -90, 0, -90, 90])
        target_pose = target_pose + self.offset
        # Move above the target pose
        self.get_logger().info(f"self.indy.movel(target_pose + np.array([0, 0, 50, 0, 0, 0]))")
        self.move_robot(target_pose + np.array([0, 0, 50, 0, 0, 0]))
        time.sleep(3)

        # # Set to custom control mode
        self.indy.set_custom_control_mode(1)
        self.get_logger().info(f"Custom control mode: {self.indy.get_custom_control_mode()}")

        self.set_control_gain_low()

        # self.get_logger().info(f"self.indy.movel(target_pose + np.array([0, 0, 50, 0, 0, 0]))")
        # self.indy.movel(target_pose + np.array([0, 0, 30, 0, 0, 0]))
        # time.sleep(3)

        # self.get_logger().info(f"self.indy.movel(target_pose + np.array([0, 0, 50, 0, 0, 0]))")
        # self.indy.movel(target_pose + np.array([0, 0, 20, 0, 0, 0]))
        # time.sleep(3)

        # self.get_logger().info("10")
        self.move_robot(target_pose + np.array([0, 0, 10, 0, 0, 0]))

        self.move_robot(target_pose + np.array([0, 0, 5, 0, 0, 0]))

        # # BC policy

        # self.get_ft_bias()

        # # Get current eef pose
        # control_data = self.indy.get_control_data()
        # curr_eef_pose = control_data['p']
        # curr_eef_vel = control_data["pdot"]
        # rot = Rot.from_euler("xyz", curr_eef_pose[3:])
        
        # # Get current ft sensor data
        # force_data = self.indy.get_ft_sensor_data()
        # force =([force_data['ft_Fx'], force_data['ft_Fy'], force_data['ft_Fz']])
        # moment = ([force_data['ft_Tx'], force_data['ft_Ty'], force_data['ft_Tz']])
        
        # # Transform force and moment to spatial frame
        # force_spatial = rot.as_matrix().dot(np.array(force))
        # moment_spatial = rot.as_matrix().dot(np.array(moment))
        # eef_wrench = np.concatenate([force_spatial, moment_spatial])


        # self.BC_policy.predict_gains(curr_eef_pose, curr_eef_vel, eef_wrench, target_pose)

        # pass

    def set_control_gain_low(self):
        gain0 = [100, 100, 100,100, 100, 100] #kp
        gain1 = [20, 20, 20, 20, 20, 20] #kv
        gain2 = [0, 0, 0, 0, 0, 0] #ki
        # Impedance Control (Task space)
        xy_gain = 1000; z_gain = 300; rot_gain = 1000

        gain3 = [xy_gain, xy_gain, z_gain, rot_gain, rot_gain, rot_gain] #kp
        xy_d_gain = np.sqrt(xy_gain) * 3
        z_d_gain = np.sqrt(z_gain) * 3
        rot_d_gain = np.sqrt(rot_gain) * 3
        gain4 = [xy_d_gain, xy_d_gain, z_d_gain, rot_d_gain, rot_d_gain, rot_d_gain] #kv
        # gain5 = [0, 0, 0, 0, 0, 0] #ki
        gain5 = [10, 10, 10, 10, 10, 10] #ki

        # Rest values
        gain6 = [0, 0, 0, 0, 0, 0] #K
        gain7 = [0, 0, 0, 0, 0, 0] #KC"
        gain8 = [0, 0, 0, 0, 0, 0] #KD
        gain9 = [0, 0, 0, 0, 0, 0] #rate
        gainzero = [0, 0, 0, 0, 0, 0]

        self.indy.set_custom_control_gain(gain0=gain0, gain1=gain1, gain2=gain2, gain3=gain3, gain4=gain4, gain5=gain5, gain6=gain6, gain7=gain7, gain8=gain8, gain9=gain9)
        self.indy.get_custom_control_gain()
    

    def move_robot(self, pose, compliant = False):
        start_time = time.time()
        timeout = 10
        while (time.time() - start_time < timeout):
            self.indy.movel(pose)
            current_p = self.indy.get_control_data()['p']

            error_x = np.array(current_p[:3]) - np.array(pose[:3])

            rotm = Rot.from_euler('xyz',current_p[3:],degrees=True).as_matrix()
            Rd = Rot.from_euler('xyz',pose[3:],degrees=True).as_matrix()

            rot_err = np.linalg.norm(np.eye(3) - Rd.T @ rotm)
            
            # print('In move_robot_task:', i, 'Time step:', time.time() - start_time)

            if np.linalg.norm(error_x, 2) < 1 and rot_err < 0.01:
                print('Task finished')
                self.indy.stop_motion(2)
                break

    def gripper_command(self, command):
        assert command in ["open", "close"], "Invalid command"
        if command == "open":
            signal = [{'port': 'C', 'states': [EndtoolState.LOW_PNP]}]
            self.indy.set_endtool_do(signal)

        elif command == "close":
            signal = [{'port': 'C', 'states': [EndtoolState.HIGH_PNP]}]
            self.indy.set_endtool_do(signal)

def main():
    rclpy.init()
    pipeline = FullPipeline()
    # future = pipeline.get_edf("pick")
    # rclpy.spin_until_future_complete(pipeline, future)
    # response = future.result()

    # pose = response.position + response.orientation # assuming that pose and orientation are both lists
    # pipeline.get_logger().info(f"pick pose: {pose}")
    # pipeline.gripper_command("open")
    # pipeline.exec_pick(pose)
    # time.sleep(5)
    # pipeline.gripper_command("close")
    # time.sleep(5)
    future = pipeline.get_edf("place")
    rclpy.spin_until_future_complete(pipeline, future)
    response = future.result()
    pose = response.position + response.orientation # assuming that pose and orientation are both lists
    pipeline.get_logger().info(f"place pose: {pose}")
    pipeline.exec_place(pose)
    pipeline.gripper_command("open")



if __name__ == "__main__": 
    main()