from robot import Robot
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy 
from geometric_admittance_control import GeometricAdmittanceControl as GAC
import time
from neuromeka import IndyDCP3, TaskTeleopType, BlendingType, EndtoolState

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from camera_interfaces.msg import CompAct
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool

class RobotGAC(Robot):
    
    def __init__(self, robot_ip, act_horizon=15):
        super().__init__(robot_ip, act_horizon = act_horizon)
        dt = 0.005 # 200 Hz
        self.initial_pose = np.array(self.robot.get_control_data()['p'])
        self.gac = GAC(self.robot, initial_pose=self.initial_pose, dt=dt)
        self.actions = self.action_init_real

        gripper_state = self.robot.get_endtool_do()['signals'][0]['states'][0]
        self.gripper_state = 0 if gripper_state == EndtoolState.LOW_PNP else 1

        self.exec_loop = self.create_timer(dt, self.exec_traj, callback_group=ReentrantCallbackGroup())
        self.act_sub = self.create_subscription(CompAct, "/indy/act2", self.get_act_comp, 10, callback_group=ReentrantCallbackGroup())
        self.bias_sub = self.create_subscription(Bool, "/indy/rebias", self.rebias_cb, 10, callback_group=ReentrantCallbackGroup()) 
        # self.mode_sub = self.create_subscription(String, '/indy/mode', self.update_gain, 10, callback_group=ReentrantCallbackGroup())
        # self.gain_pub = self.create_publisher(Twist, '/indy/gain', 10)
        self.Kp = np.eye(3) * 1000
        self.KR = np.eye(3) * 1000
        self.rebias_raised = False
        print("TELEOP GAC READY")

    def rebias_cb(self, msg):
        if msg.data: 
            print("Recollecting FT Bias")
            # self.gac.ft_bias = self.gac.measure_ft_bias()
            self.rebias_raised = True

    def get_act_comp(self, msg,debug=False): 
        px = msg.cart_pose.linear.x
        py = msg.cart_pose.linear.y
        pz = msg.cart_pose.linear.z
        ax = msg.cart_pose.angular.x
        ay = msg.cart_pose.angular.y
        az = msg.cart_pose.angular.z
        
        gpx = msg.ad_gains.linear.x
        gpy = msg.ad_gains.linear.y
        gpz = msg.ad_gains.linear.z
        grx = msg.ad_gains.angular.x
        gry = msg.ad_gains.angular.y
        grz = msg.ad_gains.angular.z
        self.Kp = np.eye(3) * np.array([gpx, gpy, gpz])
        self.KR = np.eye(3) * np.array([grx, gry, grz])

        if not debug:
            self.gac.update_gains(self.Kp, self.KR)
            self.grip(msg.gripper_state)
            self.actions = np.array([px, py, pz, ax, ay, az]).reshape((1,6))
        # print(f"Action: {[px, py, pz, ax, ay, az]}")
        # print(f"Gains: {[gpx, gpy, gpz, grx, gry, grz]}")

    def grip(self, state):
        if state == self.gripper_state:
            return
        else: 
            if state == 0:
                self.robot.set_endtool_do([{'port': 'C', 'states': [EndtoolState.LOW_PNP]}])
                self.gripper_state = 0
            elif state == 1:
                self.robot.set_endtool_do([{'port': 'C', 'states': [EndtoolState.HIGH_PNP]}])
                self.gripper_state = 1
            else:
                print("Invalid gripper state") 

    def exec_traj(self, horizon = 1, debug=False): 
        if self.rebias_raised:
            print("Recollecting FT Bias")
            self.gac.ft_bias = self.gac.measure_ft_bias()
            self.rebias_raised = False

        if horizon == 0: 
            horizon = self.act_horizon

        for i in range(horizon): 
            if debug:
                print("Moving to: ", self.actions[i])
            else:
                self.move_to(self.actions[i])
    
    def move_to(self, pose, vel_ratio=0.25, acc_ratio=4.0): # Default value is vel_ratio=0.25, acc_ratio = 4.0
        
        if hasattr(self, 'gac'):

            # pose2 = np.array([552.10, 15.13, 445.07, -180, 0, 90])

            indy_cmd_gac = self.gac.calculate_control(pose)
            # indy_cmd_gac = pose
            # print("GAC Command: ", indy_cmd_gac)
            # print(f"Target Pose: {pose}, and current pose: {self.robot.get_control_data()['p']}")
            self.robot.movetelel_abs(tpos=indy_cmd_gac, vel_ratio=vel_ratio, acc_ratio=acc_ratio)
        else:
            # If gac isn't initialized yet, call the parent move_to to avoid an error
            super().move_to(pose, vel_ratio=0.2, acc_ratio=3.0)
            time.sleep(3)
        return
    
    def cart2se3(self, p): 
        p = np.array(p)
        rot = R.from_euler('xyz', p[3:]).as_matrix()
        homog_pose = np.eye(4)
        homog_pose[:3,:3] = rot
        homog_pose[:3,3] = p[:3]/1000
        return homog_pose

    def get_rel_twist(self, pose1, pose2): 
        g_rel = np.linalg.inv(pose1) @ pose2
        twist, _ = self.se3_logmap(g_rel, type="rotmat")
        return twist                 

    def se3_logmap(self, pose, type="euler"): 
        """
        Convert a pose from the robot to a logmap representation
        """
        if type == "euler":
            pose = np.array(pose)
            p = pose[:3]
            rot = R.from_euler('xyz', pose[3:]).as_matrix()
        else: 
            rot = pose[:3,:3]
            p = pose[:3,3]
            
        w, theta = self.so3_logmap(rot)
        if theta == 0: 
            twist = np.concatenate((w*theta, p), axis = 0)
            return twist, np.eye(3)
        w_hat = self.hat_map(w, type="ang")
        # V = np.eye(3) +((1 - np.cos(theta))/(theta**2)) * w_hat + ((theta - np.sin(theta))/(theta**3)) * (w_hat @ w_hat)
        V = np.eye(3) * (1/theta) - (0.5 * w_hat) + ((1/theta) - (0.5 * (1/np.tan(theta/2)))) * (w_hat @ w_hat)
        v = V @ p
        rot = R.from_matrix(rot)
        twist = np.concatenate((rot.as_rotvec(), v*theta), axis = 0)
        return twist, V        
            
    def hat_map(self,w, type="ang"):         
        if type == "ang": 
            w_hat = np.array([[0, -w[2], w[1]],
                                [w[2], 0, -w[0]],
                                [-w[1], w[0], 0]])
        elif type == "twist":
            t = w[3:]
            w_hat = np.array([[0, -w[2], w[1], t[0]],
                                [w[2], 0, -w[0],t[1]],
                                [-w[1], w[0], 0, t[2]], 
                                [0, 0, 0, 0]])
        return w_hat 
            
    def so3_logmap(self,rot): 
        """
        Convert a rotation from the robot to a logmap representation
                """
        if np.sum(np.eye(3) - rot) < 1e-3:
            return np.zeros(3), 0
        r = R.from_matrix(rot)
        test_theta = np.arccos(np.clip((np.trace(r.as_matrix()) - 1) / 2, -1, 1))
        test_omega  = (1/(2*np.sin(test_theta))) * np.array([rot[2, 1] - rot[1, 2],rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
        return test_omega, test_theta         
    
    def se3_expmap(self, twist): 
        twist_hat = self.hat_map(twist, type="twist")
        g = scipy.linalg.expm(twist_hat)
        return g
    
    def g2command(self, g):
        """
        Convert a pose from the robot to an euler representation
        """
        p = g[:3,3] * 1000
        rot = g[:3,:3]
        r = R.from_matrix(rot)
        euler = r.as_euler('xyz', degrees=True)
        return np.concatenate((p, euler), axis=0)
    
    def update_gain(self, msg):
        # Kp and KR should be 3x3 matrices
        if msg.data == "Free":
            self.Kp = np.eye(3) * 1000
            self.KR = np.eye(3) * 1000
            self.gac.update_gains(self.Kp, self.KR)
            
        elif msg.data == "Contact":
            self.Kp = np.eye(3) * np.array([1500, 1500, 500])
            self.KR = np.eye(3) * 1500
            self.gac.update_gains(self.Kp, self.KR)

        elif msg.data == "Insertion":
            self.Kp = np.eye(3) * np.array([300, 300, 1500])
            self.KR = np.eye(3) * 300
            self.gac.update_gains(self.Kp, self.KR)

        elif msg.data == "Compliant":
            self.Kp = np.eye(3) * 500
            self.KR = np.eye(3) * 500
            self.gac.update_gains(self.Kp, self.KR)



def main():
    rclpy.init()

    robot_ip = "10.208.112.143"

    try:
        robot = RobotGAC(robot_ip)
        rclpy.spin(robot)
    except KeyboardInterrupt: 
        print("Shutting Down")
        robot.robot.stop_teleop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()