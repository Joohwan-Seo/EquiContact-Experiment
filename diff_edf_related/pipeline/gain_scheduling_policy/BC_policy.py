import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from scipy.spatial.transform import Rotation as Rot
from copy import deepcopy
import time
# from adjust import Tuner

class BCPolicy(nn.Module):
    def __init__(self, hidden_sizes, input_size=18, output_size=6, residual = False):
        super().__init__()

        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Tanh()

        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i),fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)

        self.residual = residual

    def forward(self, input):
        residual_ = False
        h = input
        for i,fc in enumerate(self.fcs):
            if self.residual:
                if i % 2 == 1:
                    feedforward = h
                    residual_ = True

            if residual_:
                h = fc(h)
                h += feedforward
                residual_ = False
            else:
                h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output
    
class BCInference:
    def __init__(self, 
                 model_path = "/home/horowitzlab/ros2_ws/src/pipeline/pipeline/BC_policy/BC_policy_GIC_GCEV_itr_39.pkl", 
                 scaling_params_path = "/home/horowitzlab/ros2_ws/src/pipeline/pipeline/BC_policy/res_2_128_2,3,4,5,6_scaling_params.json",
                 ):
        hidden_sizes = [128] * 6  
        input_size = 18
        output_size = 6
        residual = True
        self.model = BCPolicy(hidden_sizes=  hidden_sizes, input_size = input_size, output_size = output_size, residual = residual)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with open(scaling_params_path, "r") as f:
            self.scaling_params = json.load(f)


    def euler_to_rotation_matrix(self, euler): # euler: [roll, pitch, yaw] in np.array with shape [3,]
        rot = Rot.from_euler('xyz', euler, degrees=False)
        return rot.as_matrix()

    def skew_to_vec(self, S):
        return np.array([S[2,1], S[0,2], S[1,0]])

    def to_gcev(self, effector_pose, target_pose):
        p = np.array(effector_pose[:3])
        rot_euler = np.array(effector_pose[3:])
        p_d = np.array(target_pose[:3])
        rot_euler_d = np.array(target_pose[3:])

        R_mat = self.euler_to_rotation_matrix(rot_euler)
        R_d_mat = self.euler_to_rotation_matrix(rot_euler_d)

        # Position Error ep
        e_p = R_mat.T @ (p - p_d)

        # Orientation error
        e_R = self.skew_to_vec(R_d_mat.T @ R_mat - R_mat.T @ R_d_mat)

        e_G = np.concatenate([e_p, e_R])
        return e_G # size [6,]

    def v_spatial_to_body(self, linear_velocity_spatial, effector_pose):
        R_mat = self.euler_to_rotation_matrix(effector_pose[3:])
        return R_mat.T @ linear_velocity_spatial

    def pre_process(self, eef_pose, eef_vel, eef_wrench, target_pose, ft_bias):

        # check if eef_pose, eef_vel, eef_wrench, target_pose are all in numpy array
        assert isinstance(eef_pose, np.ndarray), "eef_pose should be a numpy array"
        assert isinstance(eef_vel, np.ndarray), "eef_vel should be a numpy array"
        assert isinstance(eef_wrench, np.ndarray), "eef_wrench should be a numpy array"
        assert isinstance(target_pose, np.ndarray), "target_pose should be a numpy array"
        assert isinstance(ft_bias, np.ndarray), "ft_bias should be a numpy array"

        # check if eef_pose, eef_vel, eef_wrench, target_pose are all in correct shape
        assert eef_pose.shape == (6,), "eef_pose should be of shape (6,)"
        assert eef_vel.shape == (6,), "eef_vel should be of shape (6,)"
        assert eef_wrench.shape == (6,), "eef_wrench should be of shape (6,)"
        assert target_pose.shape == (6,), "target_pose should be of shape (6,)"
        assert ft_bias.shape == (6,), "ft_bias should be of shape (6,)"


        # convert mm to m
        eef_pose[:3] = eef_pose[:3] * 0.001
        target_pose[:3] = target_pose[:3] * 0.001

        # convert deg to rad
        eef_pose[3:] = np.radians(eef_pose[3:])
        target_pose[3:] = np.radians(target_pose[3:])

        # convert mm/s to m/s
        eef_vel[:3] = eef_vel[:3] * 0.001

        # convert deg/s to rad/s
        eef_vel[3:] = np.radians(eef_vel[3:])

        # convert linear velocity from spatial to body frame
        eef_vel[:3] = self.v_spatial_to_body(eef_vel[:3], eef_pose)

        # get GCEV
        e_G = self.to_gcev(eef_pose, target_pose)

        # Bias subtraction for wrench
        eef_wrench = eef_wrench - ft_bias

        # combine all the inputs
        obs = np.concatenate([e_G, eef_vel, eef_wrench])

        return obs

    def post_process(self, predicted_act):
        act_mean = np.array(self.scaling_params["act_mean"])
        act_std = np.array(self.scaling_params["act_std"])

        # denormalization of action
        act_denormalized = predicted_act * (act_std + 1e-8) + act_mean 
        act_original = np.power(10, act_denormalized) - 1e-8
        
        return act_original

    def normalize_obs(self, obs):
        obs_mean = np.array(self.scaling_params["obs_mean"])
        obs_std = np.array(self.scaling_params["obs_std"])
        norm_obs = (obs - obs_mean) / (obs_std + 1e-8)

        return norm_obs
    
    def predict_gains(self, eef_pose, eef_vel, eef_wrench, target_pose):
        obs_preprocessed = self.pre_process(eef_pose, eef_vel, eef_wrench, target_pose,)
        #print("pre_process(input): ",self.pre_process(input))

        obs_normalized = self.normalize_obs(obs_preprocessed)
        
        obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32).unsqueeze(0)

        ### Pass through self
        with torch.no_grad():
            predicted_act = self.model(obs_tensor).cpu().numpy().squeeze()  # Get normalized actions

        ### Post process
        predicted_gains = self.post_process(predicted_act)

        return predicted_gains