import rclpy 
import open3d as o3d
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from camera_interfaces.srv import Img
from pynput.keyboard import Key, Listener, KeyCode
import time
import os
import cv_bridge as cvb 
import cv2
from argparse import ArgumentParser

global init_num
init_num = 0
class ImgClientAsync(Node): 

    def __init__(self):
        super().__init__("img_client_async")
        self.client = self.create_client(Img, "get_img")
         
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Img.Request()

    def store_client_params(self, camera_id, save_dir, num_imgs, delay): 
        self.camera_id = camera_id
        self.save_dir = save_dir
        self.num_imgs = num_imgs
        self.delay = delay


    def send_request(self, camera_id, save_dir, num_imgs, delay, init_num): 
        self.req.camera_id = camera_id
        self.req.num_imgs = int(num_imgs)
        self.req.delay = delay
        self.req.save_dir = save_dir
        self.req.init_num = init_num
        return  self.client.call_async(self.req)
class CustomListener:
    def __init__(self): 
        self.init_num = 0

    def on_press(self, key, client, args): 
        if key == Key.esc: 
            print(f"Exiting")
            return False
        elif key == KeyCode(char='t'):
            future = client.send_request(args.camera_id, args.save_dir, int(args.num_imgs), float(args.delay), int(self.init_num))
            rclpy.spin_until_future_complete(client, future)
            response = future.result()
            if response.success: 
                client.get_logger().info("Getting images succeeded")
            self.init_num += 1 
            print(f"Starting file number:{self.init_num}")    
    def on_release(self,key):
        if key == Key.esc:
            return False
def main(): 
    init_num = 0
    rclpy.init()

    parser = ArgumentParser()
    parser.add_argument("--camera_id")
    parser.add_argument("--save_dir")
    parser.add_argument("--num_imgs", default=1)
    parser.add_argument("--delay", default = 0.2)
    parser.add_argument("--keyboard", default=False, action="store_true")

    args = parser.parse_args()
    
    client = ImgClientAsync()
    listener_func = CustomListener()
    listener = Listener(on_press=lambda x: listener_func.on_press(x, client, args), on_release=listener_func.on_release)
    with listener as listener:
        listener.join()
    return
    
if __name__ == "__main__": 
    main()
