from neuromeka import IndyDCP3, TaskTeleopType, BlendingType, EndtoolState
import numpy as np 
import time 
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
import rclpy
import grpc
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from camera_interfaces.msg import DiffAction, RoboData
from geometry_msgs.msg import Twist
from functools import partial

class Robot(Node): 

    def __init__(self, robot_ip, act_horizon=15): 
        super().__init__('robot')
        self.upperbound_pos = np.array([45, 45, 0]) * 1.0 # in mm
        self.lowerbound_pos = np.array([-45, -45, 0]) * 1.0 # in mm
        self.upperbound_angle = np.array([10, 10, 10]) * 1.0 ### deg
        self.lowerbound_angle = np.array([-10, -10, -10]) * 1.0 ### in deg
        self.act_horizon = act_horizon
        self.robot_ip = robot_ip
        self.robot = IndyDCP3(robot_ip)
        
        try:
            self.robot.stop_teleop()

        # Stop teleop if hasn't ended properly
        except grpc._channel._InactiveRpcError as e:
            if "TELEOP_ALREADY_STARED" in e.details():
                self.robot.stop_teleop()
                print("Teleop Stopped")
                time.sleep(5)
        
        self.robot.start_teleop(method=TaskTeleopType.ABSOLUTE)
        time.sleep(1)
        print("I am here")

        self.pose = np.zeros((6))
        self.pose_msg = Twist()
        self.ft_msg = Twist()

        # default pose
        # init_pose = np.array([558.80, 15.13, 365.07, -180, 0, 90]) # Default
        init_pose = np.array([558.80, 0, 365.07, -180, 0, 90]) # Default with platform v2
        # init_pose = np.array([558.80, -45.33, 334.25, -30, 180, 0]) # default with platform v2 30deg
        # init_pose = np.array([478.9906091239746, -17.272042110616994, 322.7776693213716, 151.5666801770288, -2.080297903095018, 175.7491233312564])
        # init_pose = np.array([558.80, -107.33, 275.25, -45, 180, 0]) # default with platform v2 30deg

        # init_pose = np.array([539.54, 59.68, 324.39, -30, -180, 0])

        ### Testing Benchmark poses
        # init_pose = np.array([533.4, -177.8, 370, -180, 0, 90]) # 21, -7 inches
        # init_pose = np.array([355.6, 0 , 370, -180, 0, 90]) # 24, 0 inches
        # init_pose = np.array([533.4, 152.4 , 370, -180, 0, 90]) # 21, 6 inches
        # init_pose = np.array([558.80, 25.4, 370, -180, 0, 90]) # 22, 1 inch
        # init_pose = np.array([558.80, -25.4, 370, -180, 0, 90]) # 22, -1 inch

        # init_pose = np.array([457.2, 0, 280, 173, 10, 90]) # for Picking default
        # init_pose = np.array([457.2, -304.8, 280, 180, 0, 90]) # for Picking -12 inch
        # init_pose = np.array([457.2, -304.8, 280, 187, -5, 90]) # pick -5 angle

        # init_pose = np.array([443.78, -136.10, 323.69, -45, -180, 0]) 

        # init_pose = np.array([558.80, 15, 365.07, -175, 10, 90]) # 
        # init_pose = np.array([565.80, -55.33, 334.25, -25, 185, 0]) # 
        # init_pose = np.array([540.80, -30.33, 334.25, -25, 185, 0]) 
        # init_pose = np.array([552.10, -86.3, 445.07, 180, 0, 90])
        
        
        x_add = np.random.uniform(low = self.lowerbound_pos[0], high=self.upperbound_pos[0])
        y_add = np.random.uniform(low = self.lowerbound_pos[1], high=self.upperbound_pos[1])
        z_add = np.random.uniform(low = self.lowerbound_pos[2], high=self.upperbound_pos[2])
        xang_add = np.random.uniform(low = self.lowerbound_angle[0], high = self.upperbound_angle[0])
        yang_add = np.random.uniform(low = self.lowerbound_angle[1], high = self.upperbound_angle[1])
        zang_add = np.random.uniform(low = self.lowerbound_angle[2], high = self.upperbound_angle[2])

        x_add = 0
        y_add = 0
        z_add = 0
        xang_add = 0
        yang_add = 0
        zang_add = 0

        R_init = R.from_euler('xyz', init_pose[3:], degrees=True).as_matrix()

        init_pose[0:3] = init_pose[0:3] + R_init @ np.array([0, 0, -80])

        self.actions = np.zeros((1,6))

        self.actions[0,0:3] = init_pose[0:3] + R_init @ np.array([x_add, y_add, z_add])
        self.actions[0,3:6] = init_pose[3:6] + np.array([xang_add, yang_add, zang_add])

        # self.actions = np.array([350, -186.5, 522.1, 180, 0, 90]).reshape((1,6))x

        self.action_init_real = self.actions
        # self.actions = (init_pose + np.array([x_add, y_add, z_add, xang_add, yang_add, zang_add])).reshape((1,6))
        
        self.move_to(self.actions[0])
        # time.sleep(1)
        self.teleop_sub = self.create_timer(1/100, self.get_pose)
        # self.pose_pub = self.create_publisher(Twist, '/indy/pose', 10)
        # self.ft_pub = self.create_publisher(Twist, '/indy/ft', 10)
        self.state_pub = self.create_publisher(RoboData, '/indy/state', 10)
        self.exec_loop = self.create_timer(1/50, self.exec_traj, callback_group=ReentrantCallbackGroup())
        self.act_sub = self.create_subscription(DiffAction, "/indy/act", self.get_act, 10, callback_group=ReentrantCallbackGroup())

    def get_pose(self): 
        msg = RoboData()
        self.pose = self.robot.get_control_data()['p']
        vel = self.robot.get_control_data()['pdot']
        ft_data = self.robot.get_ft_sensor_data()
        # print("Pose: ", self.pose)
        # print("FT: ", ft_data)
        # print("Vel: ", vel)
        msg.ppos.linear.x = self.pose[0]
        msg.ppos.linear.y = self.pose[1]
        msg.ppos.linear.z = self.pose[2]
        msg.ppos.angular.x = self.pose[3]
        msg.ppos.angular.y = self.pose[4]
        msg.ppos.angular.z = self.pose[5]
        
        msg.pvel.linear.x = vel[0]
        msg.pvel.linear.y = vel[1]
        msg.pvel.linear.z = vel[2]
        msg.pvel.angular.x =vel[3]
        msg.pvel.angular.y =vel[4]
        msg.pvel.angular.z =vel[5]

        # msg.wrench.force.x = ft_data['ft_Fx']
        # msg.wrench.force.y = ft_data['ft_Fy']
        # msg.wrench.force.z = ft_data['ft_Fz']
        # msg.wrench.torque.x = ft_data['ft_Tx']
        # msg.wrench.torque.y = ft_data['ft_Ty']
        # msg.wrench.torque.z = ft_data['ft_Tz']

        msg.ft.linear.x = ft_data['ft_Fx']
        msg.ft.linear.y = ft_data['ft_Fy']
        msg.ft.linear.z = ft_data['ft_Fz']
        msg.ft.angular.x = ft_data['ft_Tx']
        msg.ft.angular.y = ft_data['ft_Ty']
        msg.ft.angular.z = ft_data['ft_Tz']
        
        self.state_pub.publish(msg)

    def get_act(self, msg): 
        poses = []
        for i in msg.actions: 
            twist_i = i 
            pose_arr = []
            pose_arr.append(twist_i.linear.x)
            pose_arr.append(twist_i.linear.y)
            pose_arr.append(twist_i.linear.z)
            pose_arr.append(twist_i.angular.x)
            pose_arr.append(twist_i.angular.y)
            pose_arr.append(twist_i.angular.z)
            poses.append(pose_arr)
            # print(f"Received Pose: {pose_arr}")
        
        if len(poses) > 0:
            self.actions = np.array(poses)
    
    def move_to(self, pose, vel_ratio = 0.2, acc_ratio = 3.0): 
        # print("Moving to: ", pose)
        
        self.robot.movetelel_abs(tpos=pose, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
        time.sleep(0.01)
        return
    
    def exec_traj(self, horizon = 1, debug=False): 
        if horizon == 0: 
            horizon = self.act_horizon
        for i in range(horizon): 
            if debug:
                print("Moving to: ", self.actions[i])
            else:
                self.move_to(self.actions[i])
                # print("Moving to: ", self.actions)
    

def main():
    rclpy.init()

    try:
        robot = Robot('10.208.112.143') 
        rclpy.spin(robot)
    except KeyboardInterrupt: 
        print("Shutting Down")
        robot.robot.stop_teleop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
