import rclpy 
import open3d as o3d
from rclpy.node import Node 
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import Image 
from camera_interfaces.srv import Img

import matplotlib.pyplot as plt
import argparse
import time
import os
import numpy as np
import cv_bridge as cvb 
import cv2

class ImgService(Node): 

    def __init__(self, depth=False, d2c= False): 
        super().__init__("img_service")
        self.depth = depth 
        self.d2c = d2c 
        self.img_bridge = cvb.CvBridge()
        first_cb_group = ReentrantCallbackGroup()
        self.img_sub = self.create_subscription(Image, "realsense/camera/color/image_raw", lambda x: self.img_cb_lambda(x, "/camera_01"), 10, callback_group=first_cb_group)
        self.img2_sub = self.create_subscription(Image, "/camera_01/color/image_raw", lambda x: self.img_cb_lambda(x, "/camera_02"), 10, callback_group=first_cb_group)
        self.service = self.create_service(Img, "get_img", self.get_img, callback_group=first_cb_group)
        if depth and not d2c:
            self.depth_sub = self.create_subscription(Image, "camera_01/depth/image_raw", lambda x: self.depth_cb_lambda(x, "/camera_01"), 10, callback_group=first_cb_group)
            self.depth2_sub = self.create_subscription(Image, "/realsense/camera/depth/image_raw", lambda x: self.depth_cb_lambda(x, "/camera_02"), 10, callback_group=first_cb_group)
        elif depth and d2c:
            self.depth_sub = self.create_subscription(Image, "camera_01/depth_to_color/image_raw", lambda x: self.depth_cb_lambda(x, "/camera_01"), 10, callback_group=first_cb_group)
            self.depth2_sub = self.create_subscription(Image, "/realsense/camera/depth_to_color/image_raw", lambda x: self.depth_cb_lambda(x, "/camera_02"), 10, callback_group=first_cb_group)
        self.image_cam1=None
        self.depth_image_cam1=None
        self.image_cam2=None
        self.depth_image_cam2=None
    
    def get_img(self, request, response): 
        camera_id = request.camera_id
        num_imgs = request.num_imgs
        delay = request.delay
        img_path = request.save_dir
        init_num = request.init_num
        imgs = []
        depth_imgs = []
        num = 0
        img=None
        for i in range(num_imgs):
            if camera_id == "/camera_01":
                img = self.image_cam1
                depth_img = self.depth_image_cam1
            else:
                img = self.image_cam2
                depth_img = self.depth_image_cam2
            if not img is None:
                imgs.append(img)
                depth_imgs.append(depth_img)
                self.get_logger().info(f"Got image {i}")
            time.sleep(delay)
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(os.path.join(img_path, "depth_imgs"), exist_ok=True)
        os.makedirs(os.path.join(img_path, "rgb_imgs"), exist_ok = True)
        for j in range(len(imgs)):
            img_rgb = imgs[j] #self.img_sub.img_bridge.imgmsg_to_cv2(imgs[j])
            depth_img = depth_imgs[j] #self.img_sub.img_bridge.imgmsg_to_cv2(depth_imgs[j])
            self.get_logger().info(f"saving img to {os.path.join(img_path, f'img{j}.png')}")
            print(f"saving img to {os.path.join(img_path, f'img{j+init_num:02}.png')}")
            print(f"saving depth img to {os.path.join(img_path, f'depth_img{j+init_num:02}.png')}")
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # # show images
            # plt.imshow(img_rgb)
            # plt.show()

            cv2.imwrite(os.path.join(img_path, "rgb_imgs",f"img{j+init_num:02}.png" ), img_rgb)
            if self.depth:
                cv2.imwrite(os.path.join(img_path,"depth_imgs", f"depth_img{j+init_num:02}.png" ), depth_img)

        response.success = True
        self.get_logger().info(f"Got {num_imgs} images")
        return response 
    
    def img_cb_lambda(self, data, camera_id): 
        if camera_id == "/camera_01": 
            self.image_cam1 = self.img_bridge.imgmsg_to_cv2(data)
        elif camera_id == "/camera_02":
            self.image_cam2 = self.img_bridge.imgmsg_to_cv2(data)
    
    def depth_cb_lambda(self, data, camera_id): 
        if camera_id == "/camera_01": 
            self.depth_image_cam1 = self.img_bridge.imgmsg_to_cv2(data)
        elif camera_id == "/camera_02":
            self.depth_image_cam2 = self.img_bridge.imgmsg_to_cv2(data)
            

def main(): 
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", default=True, action="store_true")
    parser.add_argument("--d2c", default=False, action="store_true")
    args = parser.parse_args() 
    print("Starting service")
    img_service = ImgService(depth=False, d2c=False)
    executor = MultiThreadedExecutor()
    executor.add_node(img_service)

    try:
        img_service.get_logger().info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        img_service.get_logger().info('Keyboard interrupt, shutting down.\n')
    img_service.destroy_node()
    rclpy.shutdown()
    rclpy.shutdown()

if __name__ == "__main__": 
    main()
