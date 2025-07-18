import rclpy 
import open3d as o3d
import numpy as np
from rclpy.node import Node 
from ctypes import *
import argparse
from pathlib import Path
from collections import deque
from sensor_msgs.msg import PointCloud2, PointField


class PCDSub(Node): 

    def __init__(self, camera_id, queue_size = 3, save=False): 
        super().__init__('pcdsub')
        self.camera_id = camera_id 
        self.pcd = PointCloud2()
        self.pcd_queue = deque([])
        self.save = save
        self.queue_size = queue_size
        self.pcd_sub = self.create_subscription(PointCloud2, camera_id + '/depth_registered/points', self.get_pcd, 10)
        print(f"Initialized sub at topic: {camera_id + '/depth_registered/points'}")
        self.DUMMY_FIELD_PREFIX = '__'

                # mappings between PointField types and numpy types
        self.type_mappings = [(PointField.INT8, np.dtype('int8')),
                        (PointField.UINT8, np.dtype('uint8')),
                        (PointField.INT16, np.dtype('int16')),
                        (PointField.UINT16, np.dtype('uint16')),
                        (PointField.INT32, np.dtype('int32')),
                        (PointField.UINT32, np.dtype('uint32')),
                        (PointField.FLOAT32, np.dtype('float32')),
                        (PointField.FLOAT64, np.dtype('float64'))]
        self.pftype_to_nptype = dict(self.type_mappings)
        self.nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in self.type_mappings)

    
    def add_pcd(self, pcd): 
        if len(self.pcd_queue) < self.queue_size: 
            self.pcd_queue.append(pcd)
        elif len(self.pcd_queue) == self.queue_size: 
            self.pcd_queue.popleft()
            self.pcd_queue.append(pcd)

    def get_pcd(self, data): 
        self.pcd = data 
        self.pcd_arr = self.pointcloud2arr(self.pcd)
        o3d_pcd = o3d.geometry.PointCloud()
        self.pcd_arr_colors = np.array(self.split_rgb_field(self.pcd_arr)) 
        colors = np.stack((self.pcd_arr_colors['r'],self.pcd_arr_colors['g'],self.pcd_arr_colors['b']), axis = 1)
        self.pcd_arr_xyz = self.get_xyz_points(self.pcd_arr)
        o3d_pcd.points = o3d.utility.Vector3dVector(self.pcd_arr_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
        self.add_pcd(o3d_pcd)
        if self.save: 
            if self.camera_id == "/camera_01": 
                path = "/home/horowitzlab/ros2_ws_clean/test_pcd/test_pcd_01.pcd"
            elif self.camera_id == "/camera_02":
                path = "/home/horowitzlab/ros2_ws_clean/test_pcd/test_pcd_02.pcd"
            
            o3d.io.write_point_cloud(path, o3d_pcd)

            self.get_logger().info(f"Saved pointcloud:{path}")
    
    def fields_to_dtype(self,fields, point_step):
    
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(
                    ('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = self.pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += self.pftype_to_nptype[f.datatype].itemsize * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list
    
    def pointcloud2arr(self, msg, squeeze=True):
        

        dtype_list = self.fields_to_dtype(msg.fields, msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.frombuffer(msg.data, dtype_list)

        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (
                fname[:len(self.DUMMY_FIELD_PREFIX)] == self.DUMMY_FIELD_PREFIX)]]

        if squeeze and msg.height == 1:
            return np.reshape(cloud_arr, (msg.width,))
        else:
            return np.reshape(cloud_arr, (msg.height, msg.width))
    
    def split_rgb_field(self, cloud_arr):

        rgb_arr = cloud_arr['rgb'].copy()
        rgb_arr.dtype = np.uint32
        r = np.asarray((rgb_arr >> 16) & 255, dtype=np.uint8)
        g = np.asarray((rgb_arr >> 8) & 255, dtype=np.uint8)
        b = np.asarray(rgb_arr & 255, dtype=np.uint8)

        # create a new array, without rgb, but with r, g, and b fields
        new_dtype = []
        for field_name in cloud_arr.dtype.names:
            field_type, field_offset = cloud_arr.dtype.fields[field_name]
            if not field_name == 'rgb':
                new_dtype.append((field_name, field_type))
        new_dtype.append(('r', np.uint8))
        new_dtype.append(('g', np.uint8))
        new_dtype.append(('b', np.uint8))
        new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

        # fill in the new array
        for field_name in new_cloud_arr.dtype.names:
            if field_name == 'r':
                new_cloud_arr[field_name] = r
            elif field_name == 'g':
                new_cloud_arr[field_name] = g
            elif field_name == 'b':
                new_cloud_arr[field_name] = b
            else:
                new_cloud_arr[field_name] = cloud_arr[field_name]
        return new_cloud_arr
    
    def get_xyz_points(self, cloud_array, remove_nans=True, dtype=float):

        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & \
                np.isfinite(cloud_array['y']) & \
                np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']

        return points

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", default = "/camera_01")
    args = parser.parse_args()
    rclpy.init()
    pcd_sub = PCDSub(args.camera_id, save=True)
    rclpy.spin(pcd_sub)
    pcd_sub.destroy_node()
    rclpy.shutdown()

main()