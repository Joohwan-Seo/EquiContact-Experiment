import torch
import numpy as np 
from preprocess import pcd2pt, format_pose
import os 
import yaml

def create_yaml(data, filename): 
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def create_metadata_file(type,filename): 
    if type == "demo": 
        data = {"__type__":"DemoSequence", "name":""}
        create_yaml(data, filename)
    elif type=="step_0":
        data = {"__type__": "TargetPoseDemo", "name":"pick"}
        create_yaml(data, filename)
    elif type=="step_1": 
        data = {"__type__": "TargetPoseDemo", "name":"place"}
        create_yaml(data, filename)
    elif type =="pcd": 
        data = {"__type__": "PointCloud", "name": "", "unit_length": "1 [m]"}
        create_yaml(data, filename)
    elif type == "pose": 
        data = {"__type__": "SE3", "name": '',"unit_length": "1 [m]"}
        create_yaml(data, filename)

def create_dataset(grasp_dir, scene_dir, grasp_pose_dir, place_pose_dir, dataset_dir):
    empty_grasp_path = "/home/horowitzlab/ros2_ws/src/camera_processing/camera_processing/pyorbbec/point_clouds/gripper_comb.pcd"
    grasp_paths = os.listdir(grasp_dir)
    scene_paths = os.listdir(scene_dir)
    grasp_pose_paths = os.listdir(grasp_pose_dir)
    place_pose_paths = os.listdir(place_pose_dir)
    grasp_paths.sort()
    scene_paths.sort()
    grasp_pose_paths.sort()
    place_pose_paths.sort()
    os.makedirs(dataset_dir, exist_ok=True)
    data_dir = os.path.join(dataset_dir, "data")
    os.makedirs(data_dir,exist_ok=True)
    for i in range(len(grasp_paths)):
        demo_i_dir = os.path.join(data_dir, "demo_"+str(i))
        # print('==========================================')
        # print(i)
        # print(f"Current grasp pcd path:{grasp_paths[i]}")
        # print(f"Current grasp pose path:{grasp_pose_paths[i]}")
        # print(f"Current place pcd path:{scene_paths[i]}")
        # print(f"Current place pose path:{place_pose_paths[i]}")
        os.makedirs(demo_i_dir, exist_ok=True)
        create_metadata_file("demo", os.path.join(demo_i_dir, "metadata.yaml"))
        os.makedirs(os.path.join(demo_i_dir, "step_0/grasp_pcd"), exist_ok=True)
        os.makedirs(os.path.join(demo_i_dir, "step_0/scene_pcd"), exist_ok=True)
        os.makedirs(os.path.join(demo_i_dir, "step_0/target_poses"), exist_ok=True)
        os.makedirs(os.path.join(demo_i_dir, "step_1/grasp_pcd"), exist_ok=True)
        os.makedirs(os.path.join(demo_i_dir, "step_1/scene_pcd"), exist_ok=True)
        os.makedirs(os.path.join(demo_i_dir, "step_1/target_poses"), exist_ok=True)
        create_metadata_file("step_0",os.path.join(demo_i_dir, "step_0/metadata.yaml") )
        create_metadata_file("step_1",os.path.join(demo_i_dir, "step_1/metadata.yaml") )
        create_metadata_file("pcd",os.path.join(demo_i_dir, "step_0/grasp_pcd/metadata.yaml"))
        create_metadata_file("pcd",os.path.join(demo_i_dir, "step_1/grasp_pcd/metadata.yaml"))
        create_metadata_file("pcd",os.path.join(demo_i_dir, "step_0/scene_pcd/metadata.yaml"))
        create_metadata_file("pcd",os.path.join(demo_i_dir, "step_1/scene_pcd/metadata.yaml"))
        create_metadata_file("pose",os.path.join(demo_i_dir, "step_0/target_poses/metadata.yaml"))
        create_metadata_file("pose",os.path.join(demo_i_dir, "step_1/target_poses/metadata.yaml"))
        pcd2pt(empty_grasp_path,os.path.join(demo_i_dir, "step_0/grasp_pcd/points.pt"),os.path.join(demo_i_dir, "step_0/grasp_pcd/colors.pt"))
        pcd2pt(os.path.join(scene_dir,scene_paths[i]),os.path.join(demo_i_dir, "step_0/scene_pcd/points.pt"),os.path.join(demo_i_dir, "step_0/scene_pcd/colors.pt"), scene=True)
        format_pose(os.path.join(grasp_pose_dir, grasp_pose_paths[i]),os.path.join(demo_i_dir, "step_0/target_poses/poses.pt"))

        pcd2pt(os.path.join(grasp_dir, grasp_paths[i]),os.path.join(demo_i_dir, "step_1/grasp_pcd/points.pt"),os.path.join(demo_i_dir, "step_1/grasp_pcd/colors.pt"))
        pcd2pt(os.path.join(scene_dir,scene_paths[i]),os.path.join(demo_i_dir, "step_1/scene_pcd/points.pt"),os.path.join(demo_i_dir, "step_1/scene_pcd/colors.pt"), scene=True)
        format_pose(os.path.join(place_pose_dir, place_pose_paths[i]),os.path.join(demo_i_dir, "step_1/target_poses/poses.pt"))
if __name__=="__main__": 
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp_dir")
    parser.add_argument("--scene_dir")
    parser.add_argument("--grasp_pose_dir")
    parser.add_argument("--place_pose_dir")
    parser.add_argument("--output_dir")
    

    args = parser.parse_args()
    create_dataset(args.grasp_dir, args.scene_dir, args.grasp_pose_dir, args.place_pose_dir, args.output_dir)