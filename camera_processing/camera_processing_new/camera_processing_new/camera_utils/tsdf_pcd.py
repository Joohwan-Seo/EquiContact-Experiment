## ROS imports
import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from camera_interfaces.srv import Img
from camera_processing_new.camera_utils.get_img_client import ImgClientAsync

## Image Processing imports
import open3d as o3d 
import numpy as np
import cv2
import os 


class TSDF_Merger(Node): 

    def __init__(self): 
        super().__init__("tsdf_merger")
        self.img_client = ImgClientAsync()
        self.imgs = []

    def tsdf(self, intrinsics, extrinsics, rgbd_imgs):
        self.volume =  o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.0005,
            sdf_trunc=0.005,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for i in range(len(extrinsics)): 
            color = rgbd_imgs[i][0]
            depth = rgbd_imgs[i][1] 
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_trunc=1.5, convert_rgb_to_intensity=False)
            # frustum_block_coords = self.volume.compute_unique_block_coordinates(
            # depth, depth_intrinsics[i], extrinsics[i], 1000,
            # 3300)
            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                1280, 720, intrinsics[i]),
                np.linalg.inv(extrinsics[i]))
            
        pcd_result = self.volume.extract_point_cloud() 
        cl, ind = pcd_result.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
        pcd_result = pcd_result.select_by_index(ind)
        o3d.visualization.draw_geometries([pcd_result], "TSDF Point Cloud")
    
    def read_imgs(self, save_dir, num_imgs): 
        rgb_paths = sorted(os.listdir(os.path.join(save_dir, "rgb_imgs")))
        depth_paths = sorted(os.listdir(os.path.join(save_dir, "depth_imgs")))
        for i in range(num_imgs):
            print(f"Reading {rgb_paths[i]} and {depth_paths[i]}") 
            rgb_img = o3d.io.read_image(os.path.join(save_dir, "rgb_imgs",rgb_paths[i]))
            depth_img = o3d.io.read_image(os.path.join(save_dir, "depth_imgs",depth_paths[i]))
            self.imgs.append((rgb_img, depth_img)) 

    def request_image(self, camera_id, save_dir, num_imgs = 1, delay = 0.2): 
        future = self.img_client.send_request(camera_id, os.path.join(save_dir), num_imgs, delay)
        rclpy.spin_until_future_complete(self.img_client, future)
        response = future.result()
        if response.success: 
            self.get_logger().info("Getting images succeeded")
        
    
    def get_tsdf(self, intrinsics, extrinsics, save_dir): 
        self.request_image("/camera_01", os.path.join(save_dir, "camera_01"))
        self.request_image("/camera_02", os.path.join(save_dir, "camera_02"))
        self.read_imgs(os.path.join(save_dir, "camera_01"), 1)
        self.read_imgs(os.path.join(save_dir, "camera_02"), 1)
        self.tsdf(intrinsics, extrinsics, self.imgs)


def main():
    rclpy.init()
    tsdf_merger = TSDF_Merger()
    extrinsics = [ np.load("/home/horowitzlab/ros2_ws/test_pcd/cad_icp_camera_01_square_v2.npy"),np.load("/home/horowitzlab/ros2_ws/test_pcd/cad_icp_camera_02_square_v3.npy")]
    save_dir = "/home/horowitzlab/ros2_ws/src/tsdf_imgs/"
    intrinsics = [np.load("/home/horowitzlab/ros2_ws/src/camera_processing_old/camera_processing/CalibrationVal/intrinsics/camera_01_1280_720.npy"),np.load("/home/horowitzlab/ros2_ws/src/camera_processing_old/camera_processing/CalibrationVal/intrinsics/camera_02_1280_720.npy")]
    tsdf_merger.get_tsdf(intrinsics, extrinsics, save_dir)
    rclpy.spin(tsdf_merger)
    rclpy.shutdown()

if __name__ == "__main__": 
    main()
