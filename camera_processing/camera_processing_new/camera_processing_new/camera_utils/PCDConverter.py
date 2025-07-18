import rclpy 
import open3d as o3d
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2, PointField
from camera_interfaces.srv import PCD
import argparse
import time
import os
import numpy as np

class PCDConverter:

    def __init__(self): 
        ############ PCD Conversion variables ############
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
        self.num=0

    def pcdmsg2pcd(self, data): 
        pcd = data 
        pcd_arr = self.pointcloud2arr(pcd)
        o3d_pcd = o3d.geometry.PointCloud()
        pcd_arr_colors = np.array(self.split_rgb_field(pcd_arr))
        colors = np.stack((pcd_arr_colors['r'],pcd_arr_colors['g'],pcd_arr_colors['b']), axis = 1)
        pcd_arr_xyz = self.get_xyz_points(pcd_arr)
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd_arr_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
        return o3d_pcd
        
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