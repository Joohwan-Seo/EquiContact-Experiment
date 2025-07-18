import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from scipy.linalg import expm
from scipy.linalg import block_diag

class GeometricAdmittanceControl:
    def __init__(self, indy, initial_pose, dt, filter_FT = True):
        self.indy = indy
        self.ft_collect_time = 2 # 2 seconds

        self.dt = dt

        self.gd_command = np.eye(4)
        self.gd_command[:3, 3] = initial_pose[:3] * 0.001 # mm to m
        self.gd_command[:3, :3] = R.from_euler("xyz", initial_pose[3:], degrees=True).as_matrix() # to rotation matrix

        self.M = np.eye(6) * np.array([1, 1, 1, 1, 1, 1]) * 0.6

        self.Kp = np.eye(3) * np.array([1000, 1000, 1000]) # 1000 N/m
        self.KR = np.eye(3) * 1000

        self.damping_ratio = 1.5

        Kt = block_diag(self.Kp, self.KR)
        self.Kd = 2 * np.sqrt(Kt) * self.damping_ratio

        self.filter_FT = filter_FT
        if self.filter_FT:
            self.Fe_filtered = np.zeros((6,))
            self.Fe_tau = 0.05

            self.alpha = self.dt / (self.Fe_tau)

        # self.Kd = np.eye(6) * 350

        self.ft_bias = self.measure_ft_bias()

    def update_gains(self, Kp, KR):
        self.Kp = Kp
        self.KR = KR
        Kt = block_diag(self.Kp, self.KR)
        self.Kd = 2 * np.sqrt(Kt) * self.damping_ratio 
        # print(f"Gains updated. Kp: {np.diag(self.Kp)}, KR: {np.diag(self.KR)}")

    def measure_ft_bias(self):
        ### FT Bias
        start = time.time()

        arr = []
        print(f'Collecting FT Bias for {self.ft_collect_time} seconds')

        ### Collect FT Bias
        while time.time() - start < self.ft_collect_time :
            sensor_data = self.indy.get_ft_sensor_data()
            sensor = [sensor_data['ft_Fx'], sensor_data['ft_Fy'], sensor_data['ft_Fz'],
                      sensor_data['ft_Tx'], sensor_data['ft_Ty'],sensor_data['ft_Tz']]
            arr.append(sensor)

            time.sleep(0.01)

        ft_bias = np.mean(arr, axis=0)

        if self.filter_FT:
            self.Fe_filtered = np.array(sensor) - ft_bias

        return ft_bias # np.array (6,)
    
    def process_ft_data(self, force_data, Re):
        """
        Process the force data to get the force in the body frame.
        """
        # Convert force data to numpy array
        Fe_raw = np.array([force_data['ft_Fx'], force_data['ft_Fy'], force_data['ft_Fz'],
                      force_data['ft_Tx'], force_data['ft_Ty'],force_data['ft_Tz']])
        # Convert to body frame
        Fe_unbiased = Fe_raw - self.ft_bias

        if self.filter_FT:
            # Apply low-pass filter to the force data
            self.Fe_filtered = (1 - self.alpha) * self.Fe_filtered + self.alpha * Fe_unbiased
            Fe_unbiased = self.Fe_filtered

        ### Working Version 2025/04/02 -- No Longer Working
        # Fe_spatial = Fe_spatial * np.array([-1, 1, -1, -1, 1, -1]) # flip the sign of the force data

        

        ### After Firmware update 2025/04/28
        Fe_body = np.zeros((6,))
        transform_matrix = np.array([[0, 1, 0],
                                     [-1, 0, 0],
                                     [0, 0, 1]])
        Fe_body[:3] = transform_matrix @ Fe_unbiased[:3] # flip the sign of the force data
        Fe_body[3:] = transform_matrix @ Fe_unbiased[3:] # flip the sign of the force data

        return Fe_body    
    
    def get_sensor_data(self): 
        robot_data = self.indy.get_control_data()
        eef_pose = np.array(robot_data['p'])
        eef_vel = np.array(robot_data["pdot"])

        # print("Rotational velocity:", eef_vel[3])

        pe = eef_pose[:3] * 0.001 # mm to m
        Re = R.from_euler("xyz", eef_pose[3:], degrees= True).as_matrix() # to rotation matrix

        v_world = eef_vel[:3] * 0.001 # mm/s to m/s
        v_body = Re.T @ v_world # world to body velocity
        w_body = np.radians(eef_vel[3:]) # rad/s #NOTE(JS): CONFIRMED that this is in the body frame

        # concat v_body and w_body to get Vb, size (6,)
        Vb = np.concatenate((v_body, w_body), axis=0)

        force_data = self.indy.get_ft_sensor_data()
        Fe_body = self.process_ft_data(force_data, Re) # size (6,)

        # Fe_body = np.zeros((6,))

        if np.linalg.norm(Fe_body) < 0.001:
            print('====== WARNING: NO FORCE DATA DETECTED, REBOOT REQUIRED ======')

        return pe, Re, Vb, Fe_body
    
    def vee_map(self, mat):
        """
        Convert a skew-symmetric matrix to a vector using the vee operator.
        """
        if mat.shape == (3, 3): # SO(3)
            return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
        elif mat.shape == (4, 4): # SE(3)
            v = np.array(mat[3, 0], mat[3, 1], mat[3, 2])
            w = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
            return np.concatenate((v, w), axis=0)
        else:
            raise ValueError("Input matrix must be 3x3 or 4x4.")
        
    def hat_map(self, vec):
        """
        Convert a vector to a skew-symmetric matrix using the hat operator.
        """
        if vec.shape == (3,):
            return np.array([[0, -vec[2], vec[1]],
                             [vec[2], 0, -vec[0]],
                             [-vec[1], vec[0], 0]])
        elif vec.shape == (6,):
            v = vec[:3]
            w = vec[3:]
            return np.array([[0, -w[2], w[1], v[0]],
                             [w[2], 0, -w[0], v[1]],
                             [-w[1], w[0], 0, v[2]],
                             [0, 0, 0, 0]])
        else:
            raise ValueError("Input vector must be of size 3 or 6.")

    def calculate_control(self, desired_pose):
        # Get the current sensor data
        pe, Re, Vb, Fe = self.get_sensor_data()

        ge = np.eye(4)
        ge[:3, :3] = Re
        ge[:3, 3] = pe

        fg = np.zeros((6,))

        if desired_pose.shape == (6,):
            pd = desired_pose[:3] * 0.001
            Rd = R.from_euler("xyz", desired_pose[3:], degrees = True).as_matrix()
        elif desired_pose.shape == (4,4):
            pd = desired_pose[:3, 3]
            Rd = desired_pose[:3, :3]

        fp = Re.T @ Rd @ self.Kp @ Rd.T @ (pe - pd) # translational elastic force
        fR = self.vee_map(self.KR @ Rd.T @ Re - Re.T @ Rd @ self.KR) # rotational elastic force

        fg[:3] = fp
        fg[3:] = fR

        # Calculate the desired velocity
        Vb_desired = Vb + self.dt * np.linalg.inv(self.M) @ (- fg - self.Kd @ Vb + Fe)

        # Update the desired pose
        self.gd_command = ge @ expm(self.dt * self.hat_map(Vb_desired))

        # convert into mm and euler angle xyz form
        command_pose = np.zeros((6,))
        command_pose[:3] = self.gd_command[:3, 3] * 1000
        command_pose[3:] = R.from_matrix(self.gd_command[:3, :3]).as_euler("xyz", degrees=True)

        return command_pose
