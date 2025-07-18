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

class PCDService(Node): 

    def __init__(self, depth=True): 
        super().__init__("pcd_service")
        first_cb_group = ReentrantCallbackGroup()
        self.pcd_sub = self.create_subscription(PointCloud2, "camera_01/depth_registered/points", lambda x: self.pcd_cb_lambda(x, "/camera_01"), 10, callback_group=first_cb_group)
        self.pcd2_sub = self.create_subscription(PointCloud2, "camera_02/depth_registered/points", lambda x: self.pcd_cb_lambda(x, "/camera_02"), 10, callback_group=first_cb_group)
        self.service = self.create_service(PCD, "get_pcd", self.get_pcd, callback_group=first_cb_group)
        self.pcd_cam1 = None
        self.pcd_cam2 = None
        self.pcd_nums = {"/camera_01": 0, "/camera_02": 0}

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

    def get_pcd(self, request, response): 
        camera_id = request.camera_id
        num_pcds = request.num_pcds
        delay = request.delay
        pcd_path = request.save_dir

        pcds = []
        depth_pcds = []
        num = 0
        pcd=None
        # self.get_logger().info(f"Getting {num_pcds} images")
        for i in range(num_pcds):
            if camera_id == "/camera_01":
                pcd = self.pcd_cam1
            else:
                pcd = self.pcd_cam2
            if not pcd is None:
                pcds.append(pcd)
                self.get_logger().info(f"Got image {i}")
            time.sleep(delay)
        # os.makedirs(pcd_path, exist_ok=True)
        self.get_logger().info(f"pcd: {pcd}")
        self.get_logger().info(f"pcds: {pcds}")
        self.get_logger().info(f"len(pcds): {len(pcds)}")
        for j in range(len(pcds)):
            pcd_rgb = pcds[j] #self.pcd_sub.pcd_bridge.pcdmsg_to_cv2(pcds[j])
            # print(f"saving depth pcd to {os.path.join(pcd_path, f'depth_pcd{j}.pcd')}")
            # filename =  f"pcd{self.pcd_nums[camera_id] + j}_cam{camera_id[-1]}.pcd" 
            filename =  f"pcd{j}_cam{camera_id[-1]}.pcd" 
            self.get_logger().info(f"Saving pcd to {os.path.join(pcd_path,filename)}")
            o3d.io.write_point_cloud(os.path.join(pcd_path,filename), pcd_rgb)
            # self.num+=1 
            self.pcd_nums[camera_id] += 1
        response.success = True
        self.get_logger().info(f"Got {num_pcds} images")
        return response 
    
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
       
    def pcd_cb_lambda(self, data, camera_id):
        if camera_id == "/camera_01": 
            self.pcd_cam1 = self.pcdmsg2pcd(data)
        elif camera_id == "/camera_02": 
            self.pcd_cam2 = self.pcdmsg2pcd(data)  
        else: 
            self.pcd_cam1 = None 
            self.pcd_cam2 = None
         
    def pcd_cb(self, data): 
        self.image_cam1 = self.pcd_bridge.pcdmsg_to_cv2(data)
    def depth_cb(self, data): 
        self.depth_image_cam1 = self.pcd_bridge.pcdmsg_to_cv2(data)
    def pcd_cb2(self, data): 
        self.image_cam2 = self.pcd_bridge.pcdmsg_to_cv2(data)
    def depth_cb2(self, data): 
        self.depth_image_cam2 = self.pcd_bridge.pcdmsg_to_cv2(data)

def main(): 
    rclpy.init()
    parser = argparse.ArgumentParser()
    args = parser.parse_args() 
    print("Starting service")
    pcd_service = PCDService()
    executor = MultiThreadedExecutor()
    executor.add_node(pcd_service)

    try:
        pcd_service.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        pcd_service.get_logger().info('Keyboard interrupt, shutting down.\n')
    pcd_service.destroy_node()
    rclpy.shutdown()
    rclpy.shutdown()

if __name__ == "__main__": 
    main()
