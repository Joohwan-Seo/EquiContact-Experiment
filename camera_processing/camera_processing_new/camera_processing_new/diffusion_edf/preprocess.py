
import open3d as o3d
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
import torch
import argparse 

def pcd2pt(filename, pt_path, colors_path,device="cuda:0", scene=False): 
    device = torch.device(device)
    pcd = o3d.io.read_point_cloud(filename)
    offset = np.array([0.025, -0.025, 0])
    pcd_points = torch.tensor(np.asarray(pcd.points)) 
    pcd_colors = torch.tensor(np.asarray(pcd.colors))
    torch.save(pcd_points, pt_path)
    torch.save(pcd_colors, colors_path)

def format_pose(file_path, out_path): 
    pose_np = np.load(file_path)
    pose_np[:3] = pose_np[:3] * 0.001
    rot = Rot.from_euler('xyz',pose_np[3:],degrees=True)
    quat = rot.as_quat(scalar_first=True)
    pose_f = np.zeros((1,7))
    pose_f[0, :4] = quat 
    pose_f[0, 4:] = pose_np[:3]
    pose_tensor = torch.tensor(pose_f)
    print(pose_tensor)
    torch.save(pose_tensor, out_path)


def read_pcds(dir, output_dir): 
    pcd_paths = os.listdir(dir)
    if not os.path.isdir(os.path.join(output_dir, "colors")): 
        os.mkdir(os.path.join(output_dir, "colors"))
    if not os.path.isdir(os.path.join(output_dir, "points")): 
        os.mkdir(os.path.join(output_dir, "points"))
    points_dir = os.path.join(output_dir, "points")
    colors_dir = os.path.join(output_dir, "colors")
    for i in range(len(pcd_paths)): 
        pcd2pt(os.path.join(dir, pcd_paths[i]),os.path.join(points_dir, "demo_"+ str(i)+".pt"),os.path.join(colors_dir, "demo_"+ str(i)+".pt") )

def visualize_pcd(data_dir): 
    points = torch.load(os.path.join(data_dir, "points.pt")).cpu().detach().numpy()
    colors= torch.load(os.path.join(data_dir, "colors.pt")).cpu().detach().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points) 
    pcd.colors = o3d.utility.Vector3dVector(colors) 
    meshframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.100, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, meshframe], "Sample Point Cloud from pytorch")
    
if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    args = parser.parse_args()
    demos = os.listdir(args.dir)
    for i in demos: 
        visualize_pcd(os.path.join(args.dir, i, "step_1", "grasp_pcd"))
        visualize_pcd(os.path.join(args.dir, i, "step_1", "scene_pcd"))
