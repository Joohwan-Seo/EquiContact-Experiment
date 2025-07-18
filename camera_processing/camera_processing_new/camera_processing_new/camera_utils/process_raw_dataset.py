import torch 
import open3d as o3d 
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import os 
import glob
import pickle
import yaml


class DataProcessor: 

    def __init__(self): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scene_pcds = []
        self.grasp_pcds = []
    
    def read_dict(self, dir): 
        dict_path = [x for x in os.listdir(dir) if x.endswith(".pkl")]
        with open(os.path.join(dir, dict_path[0]), 'rb') as f: 
            data = pickle.load(f)
        self.dict = data 
    
    def pcd2pt(self, filename, pt_path, colors_path,device="cuda:0", scene = False, crop = True): 
        device = torch.device(device)
        pcd = o3d.io.read_point_cloud(filename)
        # offset = np.array([0.025, -0.025, 0])

        if scene and crop:
            min_bound = np.array([0.3, -0.2, -0.05])
            max_bound = np.array([0.7, 0.2, 0.3])

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcd = pcd.crop(bbox)


        pcd_points = torch.tensor(np.asarray(pcd.points)) 
        pcd_colors = torch.tensor(np.asarray(pcd.colors))
        torch.save(pcd_points, pt_path)
        torch.save(pcd_colors, colors_path)

    def format_pose(self, pose, out_path): 
        pose_np = np.array(pose)
        pose_np[:3] = pose_np[:3] * 0.001
        rot = Rot.from_euler('xyz',pose_np[3:],degrees=True)
        quat = rot.as_quat()
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
        pose_f = np.zeros((1,7))
        pose_f[0, :4] = quat 
        pose_f[0, 4:] = pose_np[:3] + np.array([0, -0.011, -0.01])
        pose_tensor = torch.tensor(pose_f)
        print(pose_tensor)
        torch.save(pose_tensor, out_path)

    def read_pcds(self,type, output_dir): 
        if not os.path.isdir(os.path.join(output_dir, "colors")): 
            os.mkdir(os.path.join(output_dir, "colors"))
        if not os.path.isdir(os.path.join(output_dir, "points")): 
            os.mkdir(os.path.join(output_dir, "points"))
        points_dir = os.path.join(output_dir, "points")
        colors_dir = os.path.join(output_dir, "colors")
        for k, v in self.dict.items(): 
            pcd_path_i = v[type]
            self.pcd2pt(pcd_path_i,os.path.join(points_dir, "demo_"+ str(k[-1])+".pt"),os.path.join(colors_dir, "demo_"+ str(k[-1])+".pt") )
        
    def visualize_pcd(self, data_dir): 
        points = torch.load(os.path.join(data_dir, "points.pt")).cpu().detach().numpy()
        colors= torch.load(os.path.join(data_dir, "colors.pt")).cpu().detach().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points) 
        pcd.colors = o3d.utility.Vector3dVector(colors) 
        meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.100, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, meshframe], "Sample Point Cloud from pytorch")

    def create_yaml(self, data, filename): 
        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def create_metadata_file(self, type,filename): 
        if type == "demo": 
            data = {"__type__":"DemoSequence", "name":""}
            self.create_yaml(data, filename)
        elif type=="step_0":
            data = {"__type__": "TargetPoseDemo", "name":"pick"}
            self.create_yaml(data, filename)
        elif type=="step_1": 
            data = {"__type__": "TargetPoseDemo", "name":"place"}
            self.create_yaml(data, filename)
        elif type =="pcd": 
            data = {"__type__": "PointCloud", "name": "", "unit_length": "1 [m]"}
            self.create_yaml(data, filename)
        elif type == "pose": 
            data = {"__type__": "SE3", "name": '',"unit_length": "1 [m]"}
            self.create_yaml(data, filename)

    def create_dataset(self, demo_increment, dataset_dir, ignore_keys=[]):
        empty_grasp_path = "/home/horowitzlab/ros2_ws/test_pcds3/empty_gripper.pcd"
        # grasp_paths = os.listdir(grasp_dir)
        # scene_paths = os.listdir(scene_dir)
        # grasp_pose_paths = os.listdir(grasp_pose_dir)
        # place_pose_paths = os.listdir(place_pose_dir)
        # grasp_paths.sort()
        # scene_paths.sort()
        # grasp_pose_paths.sort()
        # place_pose_paths.sort()
        os.makedirs(dataset_dir, exist_ok=True)
        data_dir = os.path.join(dataset_dir, "data")
        os.makedirs(data_dir,exist_ok=True)
        # for i in range(len(grasp_paths)):
        first_key = 0
        for k, v in self.dict.items():
            demo_i_dir = os.path.join(data_dir, "demo_"+str(first_key+demo_increment))
            # print('==========================================')
            print(k)
            # print(f"Current grasp pcd path:{grasp_paths[i]}")
            # print(f"Current grasp pose path:{grasp_pose_paths[i]}")
            # print(f"Current place pcd path:{scene_paths[i]}")
            # print(f"Current place pose path:{place_pose_paths[i]}")
            
            if k not in ignore_keys:
                scene_pcd_path = v["scene"]
                grasp_pcd_path = v["gripper"]
                grasp_pose = v["grasp"]
                place_pose = v["place"]
                os.makedirs(demo_i_dir, exist_ok=True)
                self.create_metadata_file("demo", os.path.join(demo_i_dir, "metadata.yaml"))

                os.makedirs(os.path.join(demo_i_dir, "step_0/grasp_pcd"), exist_ok=True)
                os.makedirs(os.path.join(demo_i_dir, "step_0/scene_pcd"), exist_ok=True)
                os.makedirs(os.path.join(demo_i_dir, "step_0/target_poses"), exist_ok=True)
                os.makedirs(os.path.join(demo_i_dir, "step_1/grasp_pcd"), exist_ok=True)
                os.makedirs(os.path.join(demo_i_dir, "step_1/scene_pcd"), exist_ok=True)
                os.makedirs(os.path.join(demo_i_dir, "step_1/target_poses"), exist_ok=True)

                self.create_metadata_file("step_0",os.path.join(demo_i_dir, "step_0/metadata.yaml") )
                self.create_metadata_file("step_1",os.path.join(demo_i_dir, "step_1/metadata.yaml") )
                self.create_metadata_file("pcd",os.path.join(demo_i_dir, "step_0/grasp_pcd/metadata.yaml"))
                self.create_metadata_file("pcd",os.path.join(demo_i_dir, "step_1/grasp_pcd/metadata.yaml"))
                self.create_metadata_file("pcd",os.path.join(demo_i_dir, "step_0/scene_pcd/metadata.yaml"))
                self.create_metadata_file("pcd",os.path.join(demo_i_dir, "step_1/scene_pcd/metadata.yaml"))
                self.create_metadata_file("pose",os.path.join(demo_i_dir, "step_0/target_poses/metadata.yaml"))
                self.create_metadata_file("pose",os.path.join(demo_i_dir, "step_1/target_poses/metadata.yaml"))

                self.pcd2pt(empty_grasp_path,os.path.join(demo_i_dir, "step_0/grasp_pcd/points.pt"),os.path.join(demo_i_dir, "step_0/grasp_pcd/colors.pt"))
                self.pcd2pt(scene_pcd_path,os.path.join(demo_i_dir, "step_0/scene_pcd/points.pt"),os.path.join(demo_i_dir, "step_0/scene_pcd/colors.pt"), scene=True, crop = True)
                self.format_pose(grasp_pose,os.path.join(demo_i_dir, "step_0/target_poses/poses.pt"))

                self.pcd2pt(grasp_pcd_path,os.path.join(demo_i_dir, "step_1/grasp_pcd/points.pt"),os.path.join(demo_i_dir, "step_1/grasp_pcd/colors.pt"))
                self.pcd2pt(scene_pcd_path,os.path.join(demo_i_dir, "step_1/scene_pcd/points.pt"),os.path.join(demo_i_dir, "step_1/scene_pcd/colors.pt"), scene=True, crop = True)
                self.format_pose(place_pose,os.path.join(demo_i_dir, "step_1/target_poses/poses.pt"))
            first_key+=1

if __name__ == "__main__": 
    dp = DataProcessor()
    dp.read_dict("/home/horowitzlab/ros2_ws/test_pcds3/dataset4")
    dp.create_dataset(0, "/home/horowitzlab/new_dataset3_crop", ignore_keys=['demo12'])

