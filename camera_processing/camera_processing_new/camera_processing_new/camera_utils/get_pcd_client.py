import rclpy 
import open3d as o3d
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from camera_interfaces.srv import PCD
import time
import os
import cv_bridge as cvb 
import cv2
from argparse import ArgumentParser

class PCDClientAsync(Node): 

    def __init__(self):
        super().__init__("pcd_client_async")
        self.client = self.create_client(PCD, "get_pcd")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = PCD.Request()

    def send_request(self, camera_id, save_dir, num_pcds, delay): 
        self.req.camera_id = camera_id
        self.req.num_pcds = int(num_pcds)
        self.req.delay = delay
        self.req.save_dir = save_dir

        return  self.client.call_async(self.req)
        
def main(): 
    rclpy.init()
    parser = ArgumentParser()
    parser.add_argument("--camera_id")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_pcds", default=1)
    parser.add_argument("--delay", default = 0.2)

    args = parser.parse_args()
    client = PCDClientAsync()
    future = client.send_request(args.camera_id, args.save_dir, int(args.num_pcds), float(args.delay))
    rclpy.spin_until_future_complete(client, future)
    response = future.result()
    if response.success: 
        client.get_logger().info("Getting images succeeded")