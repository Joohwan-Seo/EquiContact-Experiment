import pyspacemouse
from neuromeka import IndyDCP3, TaskTeleopType, BlendingType, EndtoolState
from pynput.keyboard import Key, Listener, KeyCode
from geometric_admittance_control import GeometricAdmittanceControl as GAC

import argparse
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import grpc # For handling exceptions
import threading

np.set_printoptions(precision=4, suppress=True)

class IndyTeleop:
    def __init__(self, args):
        
        self.indy = IndyDCP3("10.208.112.143")
        self.success = pyspacemouse.open()
        
        self.key_listener = Listener(on_press = self.on_press, on_release= self.on_release)
        self.key_listener.start()
        self.home = False
        self.rand_init = False

        self.dt = 0.005 # 200 Hz

        self.frame = args.frame
        if args.rand == "xyzuvw":
            self.upperbound_pos = np.array([45, 45, 0]) #### in mm
            self.lowerbound_pos = np.array([-45, -45, 0]) #### in mm
            self.upperbound_angle = np.array([5, 5, 5]) ### angle
            self.lowerbound_angle = np.array([-5, -5, -5]) ### angle
        else:
            self.upperbound_pos = np.array([50, 50, 0]) #### in mm
            self.lowerbound_pos = np.array([-50, -50, 0]) #### in mm
            self.upperbound_angle = np.array([0, 0, 0]) ### angle
            self.lowerbound_angle = np.array([0, 0, 0]) ### angle

    def hat_map(vec):
        # return skew-symmetric matrix from vec
        return np.array([[0, -vec[2], vec[1]],
                            [vec[2], 0, -vec[0]],
                            [-vec[1], vec[0], 0]])

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
            self.indy.set_endtool_do(signal)
        
        elif key == KeyCode(char='0'):
            print("----------Open Gripper----------")
            signal = [{'port': 'C', 'states': [EndtoolState.LOW_PNP]}]
            self.indy.set_endtool_do(signal)

        elif key == KeyCode(char='h'):
            self.home = True

        elif key == KeyCode(char='i'):
            self.rand_init = True
            
    def on_release(self, key):
        if key == Key.esc:
            print("----------Exiting----------")
            return False
        
    def teleop(self):
        print("--------------------Home Position--------------------")
        self.indy.movej([0, 0, -90, 0, -90, 90])
        time.sleep(3)
        home_pose = np.array(self.indy.get_control_state()['p'])

        TCP_d_pos = home_pose[:3] # in mm 
        TCP_d_rotm = R.from_euler('xyz',home_pose[3:],degrees=True).as_matrix() # in degrees

        self.gac = GAC(indy = self.indy, initial_pose = home_pose, dt = self.dt)
        
        speed_mode = 0 # High 0, Medium 1, Low 2
        prev_button_0 = 0
        prev_button_1 = 0

        self.indy.set_tpos_variable([{'addr': 1, 'tpos': np.zeros((6,))}])    
        
        try:
            self.indy.start_teleop(method=TaskTeleopType.ABSOLUTE)
            prev_time = time.time()
            # run every loop at 100Hz
            while True:
                
                if time.time() - prev_time < self.dt:
                    continue

                prev_time = time.time()

                ## Read the state of the SpaceMouse
                state = pyspacemouse.read()

                if self.frame == "base":
                    vel = np.array([-state.x, -state.y, state.z, state.pitch, -state.roll, -state.yaw])
                elif self.frame == "wrist":
                    vel = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])
                else:
                    vel = np.array([state.x, state.y, state.z, state.pitch, -state.roll, -state.yaw])


                ## Set offsets based on lower button
                if state.buttons[0]:
                    # Orientation Mode
                    if prev_button_0 == 1:
                        print("Orientation")
                        prev_button_0 = 0
                    pos_offset = 0.0002
                    rotation_offset = np.array([0.1,0.1,0.2])*np.pi/180
                
                else:
                    # Translation Mode
                    if prev_button_0 == 0:
                        print("Translation")
                        prev_button_0 = 1
                    pos_offset = 0.001
                    rotation_offset = np.array([0,0,0.2])*np.pi/180


                ## Set speed based on upper button
                if prev_button_1 == 0 and state.buttons[1]:
                    speed_mode += 1
                    if speed_mode > 3:
                        speed_mode = 0
                        
                    mode_messages = {
                        0: "High Speed",
                        1: "Medium Speed",
                        2: "Low Speed",
                        3: "Insertion Mode"
                    }
                    print(f"Speed Mode: {speed_mode} - {mode_messages[speed_mode]}")

                if speed_mode == 0:
                    # print("High Speed\n")
                    pass

                elif speed_mode == 1:
                    # print("Medium Speed\n")
                    pos_offset *= .5
                    rotation_offset *= .5

                elif speed_mode == 2:
                    # print("Low Speed\n")
                    pos_offset *= .1
                    rotation_offset *= .25
                    vel = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])

                elif speed_mode == 3:
                    # print("Insertion Mode\n")
                    vel = np.array([state.x, state.y, state.z, -state.pitch, state.roll, -state.yaw])
                    pos_offset *= .25
                    rotation_offset *= 0.25

                    # Convert from end-effector to spatial frame
                    g_se = self.get_se3()
                    rotm = g_se[:3,:3]
                    
                    vel[:3] = rotm @ vel[:3]
                    vel[3:] = rotm @ vel[3:]
                    temp = vel[4]
                    vel[4] = vel[3]
                    vel[3] = temp
                    vel[5] *= -1

                    vel[0] = 0.0 
                    vel[1] = 0.0
                    vel[2] *= -1 
                
                prev_button_1 = state.buttons[1]
                

                ## Delta position and orientation 
                delta_pos = vel[:3]*pos_offset *1000
                delta_ori = R.from_euler("xyz", np.multiply(vel[3:], rotation_offset)).as_matrix()

                ## Desired position and orientation
                TCP_d_pos += delta_pos
                TCP_d_rotm = TCP_d_rotm @ delta_ori 
                TCP_d_euler = R.from_matrix(TCP_d_rotm).as_euler("xyz", degrees=True)
                

                ## Stack and Move
                indy_cmd = np.hstack([TCP_d_pos, TCP_d_euler])

                if self.home:
                    self.home = False
                    TCP_d_pos = home_pose[:3] # in mm
                    TCP_d_rotm = R.from_euler('xyz',home_pose[3:],degrees=True).as_matrix()
                    
                elif self.rand_init:
                    ## Random Initialization above the hole
                    x_add = np.random.uniform(low = self.lowerbound_pos[0], high=self.upperbound_pos[0])
                    y_add = np.random.uniform(low = self.lowerbound_pos[1], high=self.upperbound_pos[1])
                    z_add = np.random.uniform(low = self.lowerbound_pos[2], high=self.upperbound_pos[2])
                    xang_add = np.random.uniform(low = self.lowerbound_angle[0], high = self.upperbound_angle[0])
                    yang_add = np.random.uniform(low = self.lowerbound_angle[1], high = self.upperbound_angle[1])
                    zang_add = np.random.uniform(low = self.lowerbound_angle[2], high = self.upperbound_angle[2])
                    init_pose = [552.10, 15.13, 445.07, -180, 0, 90]
                    rand_init_pose = init_pose + np.array([x_add, y_add, z_add, xang_add, yang_add, zang_add])
                    
                    rand_init_rotm = R.from_euler('xyz',rand_init_pose[3:],degrees=True).as_matrix()

                    target_rand_init_rotm = rand_init_rotm

                    target_rand_init = np.zeros((6,))
                    target_rand_init[:3] = rand_init_pose[:3] # Relative to home position (base frame)
                    target_rand_init[3:] = R.from_matrix(target_rand_init_rotm).as_euler("xyz", degrees=True)
                    TCP_d_pos = copy.deepcopy(target_rand_init[:3])
                    TCP_d_euler = copy.deepcopy(target_rand_init[3:])
                    TCP_d_rotm = np.eye(3)@ target_rand_init_rotm
                    self.rand_init = False

                else:
                    ## Teleoperation

                    # indy_cmd_gac = self.gac.calculate_control(indy_cmd)
                    indy_cmd_gac = indy_cmd

                    # print(indy_cmd_gac)
                    self.indy.movetelel_abs(tpos = indy_cmd_gac, vel_ratio = 0.2, acc_ratio = 0.3)
                    
                    # indy cmd (spatial frame)
                    pos = indy_cmd[:3]
                    rotm = TCP_d_rotm
                    rotm_euler = R.from_matrix(rotm).as_euler("xyz", degrees=True)
                    indy_cmd_spatial = np.hstack([pos, rotm_euler])
                    self.indy.set_tpos_variable([{'addr': 1, 'tpos': indy_cmd_spatial}])

                    tposes = self.indy.get_tpos_variable()
                    act_data =  next((item for item in tposes['variables'] if item['addr'] == 1), None)
                    act = act_data['tpos']
                

        # Ctrl+C to stop teleop
        except KeyboardInterrupt:
            self.indy.stop_teleop()

        # Stop teleop if hasn't ended properly
        except grpc._channel._InactiveRpcError as e:
            if "TELEOP_ALREADY_STARED" in e.details():
                self.indy.stop_teleop()
                print("Teleop Stopped")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, default="base", help="base or wrist")
    parser.add_argument("--rand", type=str, default="xyz", help="xyz or xyzuvw")
    args = parser.parse_args()

    indy_teleop = IndyTeleop(args)
    indy_teleop.teleop()

if __name__ == "__main__":
    main()
