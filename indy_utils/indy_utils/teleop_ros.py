import pyspacemouse
from pynput.keyboard import Key, Listener, KeyCode

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from neuromeka import IndyDCP3, TaskTeleopType, BlendingType, EndtoolState

import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist
from camera_interfaces.msg import DiffAction, CompAct
from std_msgs.msg import String

import copy

np.set_printoptions(precision=4, suppress=True)

class IndyTeleopROS(Node):
    def __init__(self, args, init_xy = None):
        super().__init__('indy_teleop_ros')
        robot_ip = "10.208.112.143"
        self.indy = IndyDCP3(robot_ip)
        self.success = pyspacemouse.open()        

        self.key_listener = Listener(on_press = self.on_press, on_release= self.on_release)
        self.key_listener.start()

        self.dt = 0.005 # 200 Hz
        self.action_msg = DiffAction()
        self.action_msg = CompAct()
        self.mod_msg = String()

        self.home = False
        self.rand_init = False
        self.init_xy = init_xy


        self.action_timer = self.create_timer(self.dt, self.teleop, callback_group=ReentrantCallbackGroup())
        self.logging_timer = self.create_timer(1/30, self.log_act, callback_group=ReentrantCallbackGroup())
        self.action_pub = self.create_publisher(CompAct, '/indy/act2', 10)
        self.logging_act_pub = self.create_publisher(CompAct, '/indy/act_logging', 10)
        # self.mod_pub = self.create_publisher(String, '/indy/mode', 10)

        self.home_pose = [350, -186.5, 522.1, 180, 0, 90] # mm and degrees
        # self.home_pose = np.array([539.54, 59.68, 324.39, -30, -180, 0])
        self.init_pose = np.array([558.80, -254, 365.07, -180, 0, 90]) # Default
        # self.init_pose = np.array([558.80, -65.33, 334.25, -30, 180, 0]) # default with platform v2 30deg
        # self.init_pose = np.array([539.54, 59.68, 324.39, -30, -180, 0]) # Data collection with tilted
        # self.init_pose = np.array([457.2, 0, 280, -180, 0, 90]) # Pick ACT 18", 0", 278mm
        # self.init_pose = np.array([538.80, -45.33, 334.25, -30, 180, 0]) # 30 degree
        # self.init_pose = np.array([531.30, -43, 295.44, -45, 180, 0]) # 45 degree

        R_init = R.from_euler('xyz', self.init_pose[3:], degrees=True).as_matrix()

        self.init_pose[0:3] = self.init_pose[0:3] + R_init @ np.array([0, 0, -80])
        # self.home_pose[0:3] = self.home_pose[0:3] + R_init @ np.array([0, 0, -80])

        self.gains = np.ones((6,)) * 1000.0
        self.gripper_state = 1
        # self.num_shooting_mode = int(1)

        self.frame = args.frame
        if args.rand == "xyzuvw":
            self.upperbound_pos = np.array([45, 45, 0]) #### in mm
            self.lowerbound_pos = np.array([-45, -45, 0]) #### in mm
            self.upperbound_angle = np.array([15, 15, 45]) ### angle
            self.lowerbound_angle = np.array([-15, -15, -45]) ### angle
        else:
            self.upperbound_pos = np.array([50, 50, 0]) #### in mm
            self.lowerbound_pos = np.array([-50, -50, 0]) #### in mm
            self.upperbound_angle = np.array([0, 0, 0]) ### angle
            self.lowerbound_angle = np.array([0, 0, 0]) ### angle

         # Set initial values for variables that persist between callbacks
        self.speed_mode = 0 # High 0, Medium 1, Low 2
        self.prev_button_0 = 0
        self.prev_button_1 = 0
        # Get home pose once at the beginning
        # self.home_pose = np.array(self.indy.get_control_state()['p'])
        self.TCP_d_pos = self.home_pose[:3]
        self.TCP_d_rotm = R.from_euler('xyz', self.home_pose[3:], degrees=True).as_matrix()      

    def hat_map(w):
        assert w.shape[0] in [3, 6], "Input must be a 3D or 6D vector"         
        if w.shape[0] == 3: 
            w_hat = np.array([[0, -w[2], w[1]],
                                [w[2], 0, -w[0]],
                                [-w[1], w[0], 0]])
        elif w.shape[0] == 6:
            t = w[3:]
            w_hat = np.array([[0, -w[2], w[1], t[0]],
                                [w[2], 0, -w[0],t[1]],
                                [-w[1], w[0], 0, t[2]], 
                                [0, 0, 0, 0]])
        return w_hat

    def get_se3(self):
        pose = self.indy.get_control_state()['p']

        g = np.eye(4)
        g[:3,:3] = R.from_euler('xyz',pose[3:],degrees=True).as_matrix()
        g[:3,3] = np.array(pose[:3]) * 0.001
        return g

    def on_press(self, key):
        if key == KeyCode(char='9'):
            print("----------Close Gripper----------")
            signal = [{'port': 'C', 'states': [EndtoolState.HIGH_PNP]}]
            self.gripper_state = 1
            self.indy.set_endtool_do(signal)
        
        elif key == KeyCode(char='0'):
            print("----------Open Gripper----------")
            self.gripper_state = 0
            signal = [{'port': 'C', 'states': [EndtoolState.LOW_PNP]}]
            self.indy.set_endtool_do(signal)

        elif key == KeyCode(char='h'):
            self.home = True

        elif key == KeyCode(char='i'):
            self.rand_init = True

        elif key == KeyCode(char='a'):
            print("=== Free Space Gain Mode ===")
            self.gains = np.ones((6,)) * 1000.0


        elif key == KeyCode(char='f'):
            print("=== Contact Gain Mode ===")
            self.gains = np.array([1500.0, 1500.0, 300.0, 1500.0, 1500.0, 1500.0])
        
        elif key == KeyCode(char='d'):
            print("=== Insertion Gain Mode ===")
            self.gains = np.array([300.0, 300.0, 1500.0, 300.0, 300.0, 300.0])
        
        elif key == KeyCode(char='s'):
            print("=== Compliant Gain Mode ===")
            self.gains = np.array([300.0, 300.0, 300.0, 300.0, 300.0, 300.0])

            
    def on_release(self, key):
        if key == Key.esc:
            print("----------Exiting----------")
            return False
        
    def teleop(self):

        ## Read the state of the SpaceMouse
        state = pyspacemouse.read()

        if self.frame == "base":
            vel = np.array([-state.x, -state.y, state.z, -state.roll, state.pitch, state.yaw])
        elif self.frame == "wrist":
            vel = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])
        else:
            vel = np.array([state.x, state.y, state.z, state.pitch, -state.roll, -state.yaw])


        ## Set offsets based on lower button
        if state.buttons[0]:
            # Orientation Mode
            if self.prev_button_0 == 1:
                print("Orientation")
                self.prev_button_0 = 0
            pos_offset = 0.0002
            rotation_offset = np.array([0.1,0.1,0.2])*np.pi/180
        
        else:
            # Translation Mode
            if self.prev_button_0 == 0:
                print("Translation")
                self.prev_button_0 = 1
            pos_offset = 0.001
            rotation_offset = np.array([0,0,0.2])*np.pi/180


        ## Set speed based on upper button
        if self.prev_button_1 == 0 and state.buttons[1]:
            self.speed_mode += 1
            if self.speed_mode > 3:
                self.speed_mode = 0
                
            mode_messages = {
                0: "High Speed",
                1: "Medium Speed",
                2: "Low Speed",
                3: "Insertion Mode"
            }
            print(f"Speed Mode: {self.speed_mode} - {mode_messages[self.speed_mode]}")

        if self.speed_mode == 0:
            # print("High Speed\n")
            g_se = self.get_se3()
            rotm = g_se[:3,:3].copy()
            vel_3_copy = copy.deepcopy(vel[3:]) # Copy the rotational velocities
            vel[3] = vel_3_copy[1]
            vel[4] = vel_3_copy[0]
            vel[5] *= -1 # Swap the first two rotational velocities
            # vel[3] = 0
            # vel[4] = 0
            # vel[5] = 0
            vel[3:] = rotm.T @ vel[3:] # Convert from end-effector to spatial frame
            pass

        elif self.speed_mode == 1:
            # print("Medium Speed\n")
            pos_offset *= .5
            rotation_offset *= .5

        elif self.speed_mode == 2:
            # print("Low Speed\n")
            pos_offset *= .1
            rotation_offset *= .25
            # vel = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])

        elif self.speed_mode == 3:
            # print("Insertion Mode\n")
            mouse_input = np.array([state.x, state.y, state.z, state.pitch, state.roll, state.yaw])
            pos_offset *= 0.5
            rotation_offset *= 0.5

            # Convert from end-effector to spatial frame
            g_se = self.get_se3()
            rotm = g_se[:3,:3].copy()

            # mouse_input[0] *= -1
            # mouse_input[1] *= -1
            tmp = mouse_input[0]
            mouse_input[0] = mouse_input[1]
            mouse_input[1] = tmp

            tmp2 = mouse_input[3]
            mouse_input[3] = mouse_input[4]
            mouse_input[4] = tmp2

            mouse_input[0] *= 1
            mouse_input[1] *= 1
            mouse_input[2] *= -1 # Invert Z for insertion

            
            mouse_input[3] *= -1
            mouse_input[4] *= -1
            mouse_input[5] *= -1
            
            trans_vel = rotm @ mouse_input[:3]
            rot_vel = rotm @ mouse_input[3:]
            
            vel = np.zeros(6)
            vel[0] = trans_vel[0]
            vel[1] = trans_vel[1]
            vel[2] = trans_vel[2]
            # vel[3] = rot_vel[0]
            # vel[4] = rot_vel[1]
            # vel[5] = rot_vel[2]
            vel[3] = mouse_input[3]
            vel[4] = mouse_input[4]
            vel[5] = mouse_input[5]

            # print(f"mouse input: {mouse_input}")
            # print(f"rotm: {rotm}")
            

            # vel = mouse_input

            # vel[0] = 0.0 
            # vel[1] = 0.0
            # vel[2] *= -1 
        
        self.prev_button_1 = state.buttons[1]
        
        
        ## Delta position and orientation 
        delta_pos = vel[:3]*pos_offset * 1000
        delta_ori = R.from_euler("xyz", np.multiply(vel[3:], rotation_offset)).as_matrix()

        ## Desired position and orientation
        self.TCP_d_pos += delta_pos
        self.TCP_d_rotm = self.TCP_d_rotm @ delta_ori 
        # self.TCP_d_rotm = delta_ori @ self.TCP_d_rotm # Apply rotation in the correct order
        self.TCP_d_euler = R.from_matrix(self.TCP_d_rotm).as_euler("xyz", degrees=True)

        # if self.speed_mode == 3:
        #     print(f"translational vel: {trans_vel}")
        #     print(f"delta_pos: {delta_pos}")
        #     print(f"Desired Position: {self.TCP_d_pos}")
        #     print(f"Desired Orientation: {self.TCP_d_euler}")
        #     print(f"===========================")


        ## Stack and Move
        self.indy_cmd = np.hstack([self.TCP_d_pos, self.TCP_d_euler])
        # print(f"Indy Command: {self.indy_cmd}")
        if self.home:
            self.home = False
            self.TCP_d_pos = self.home_pose[:3] # in mm
            self.TCP_d_rotm = R.from_euler('xyz',self.home_pose[3:],degrees=True).as_matrix()
            
        elif self.rand_init:
            ## Random Initialization above the hole
            if self.init_xy is not None:
                x_add = self.init_xy[0]
                y_add = self.init_xy[1]
                z_add = 0              

            else:
                x_add = np.random.uniform(low = self.lowerbound_pos[0], high=self.upperbound_pos[0])
                y_add = np.random.uniform(low = self.lowerbound_pos[1], high=self.upperbound_pos[1])
                z_add = 0

            # z_add = np.random.uniform(low = self.lowerbound_pos[2], high=self.upperbound_pos[2])
            xang_add = np.random.uniform(low = self.lowerbound_angle[0], high = self.upperbound_angle[0])
            yang_add = np.random.uniform(low = self.lowerbound_angle[1], high = self.upperbound_angle[1])
            zang_add = np.random.uniform(low = self.lowerbound_angle[2], high = self.upperbound_angle[2])

            # x_add = 0
            # y_add = 0
            # z_add = 0
            # xang_add = 0
            # yang_add = 0
            # zang_add = 0

            rand_init_pose = copy.deepcopy(self.init_pose)

            R_init = R.from_euler('xyz', self.init_pose[3:], degrees=True).as_matrix()
            rand_init_pose[:3] = rand_init_pose[:3] + R_init @ np.array([x_add, y_add, z_add])
            rand_init_rotm = R_init @ R.from_euler('xyz', np.array([xang_add, yang_add, zang_add]), degrees=True).as_matrix()

            rand_init_pose[3:] = R.from_matrix(rand_init_rotm).as_euler("xyz", degrees=True)
            

            self.TCP_d_pos = rand_init_pose[:3] # in mm
            self.TCP_d_euler = R.from_matrix(rand_init_rotm).as_euler("xyz", degrees=True)
            self.TCP_d_rotm = rand_init_rotm
            self.rand_init = False
            print(f"Random Init Pose: {rand_init_pose}")

        else:
            msg = Twist()
            msg.linear.x = self.indy_cmd[0]
            msg.linear.y = self.indy_cmd[1]
            msg.linear.z = self.indy_cmd[2]

            msg.angular.x = self.indy_cmd[3]
            msg.angular.y = self.indy_cmd[4]
            msg.angular.z = self.indy_cmd[5]
            
            gains_msg = Twist()
            gains_msg.linear.x = self.gains[0]
            gains_msg.linear.y = self.gains[1]
            gains_msg.linear.z = self.gains[2]

            gains_msg.angular.x = self.gains[3]
            gains_msg.angular.y = self.gains[4]
            gains_msg.angular.z = self.gains[5]

            self.action_msg.ad_gains = gains_msg 
            self.action_msg.cart_pose = msg
            self.action_msg.gripper_state = self.gripper_state
            self.action_pub.publish(self.action_msg)
            # self.logging_act_pub.publish(self.action_msg)

    def log_act(self): 
        # Log the current action
        self.logging_act_pub.publish(self.action_msg)
        # print(f"Action: {self.action_msg.cart_pose}")
        # print(f"Gains: {self.action_msg.ad_gains}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default="base", help="base or wrist")
    parser.add_argument("--rand", type=str, default="xyzuvw", help="xyz or xyzuvw")

    args = parser.parse_args()

    # init_xy = np.array([50, 50])
    # init_xy = np.array([50, -50])
    # init_xy = np.array([-50, 50])
    # init_xy = np.array([-50, -50])
    # init_xy = np.array([70, 0])
    # init_xy = np.array([0, 70])
    # init_xy = np.array([-70, 0])
    # init_xy = np.array([0, -70])

    init_xy = None

    rclpy.init()

    try:
        robot = IndyTeleopROS(args, init_xy) 
        rclpy.spin(robot)
    except KeyboardInterrupt: 
        print("Shutting Down")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
