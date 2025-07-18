import rclpy 
from rclpy.node import Node 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import cv_bridge as cvb
import cv2
import argparse
from sensor_msgs.msg import Image
from neuromeka import IndyDCP3 
from scipy.spatial.transform import Rotation as Rot
from pynput.keyboard import Key, Listener, KeyCode
import json 
import os 
import copy
import time
import sys
import numpy as np
from functools import partial

class DataLogger(Node): 

    def __init__(self, camera_ids, output_dir, depth = False, video=False):
        super().__init__('data_logger')
        self.camera_ids = camera_ids
        self.output_dir = output_dir
        self.video = video
        cb_group = ReentrantCallbackGroup()
        self.imgs = {camera_id.replace("/", ""): np.zeros((1280, 720))for camera_id in camera_ids}
        self.depth_imgs = {camera_id: np.zeros((1280, 720)) for camera_id in camera_ids}
        self.img_subs = {camera_id.replace("/", ""): self.create_subscription(Image, f"{camera_id}/color/image_raw", partial(self.img_cb, camera_id=camera_id.replace("/", "")), 10, callback_group=ReentrantCallbackGroup()) for camera_id in camera_ids}
        print(self.img_subs)
        print([f"{camera_id}/color/image_raw" for camera_id in camera_ids])
        # self.depth_subs = {self.create_subscription(Image, f"{camera_id}/depth/image_raw", lambda x: self.depth_cb(camera_id, x), 10) for camera_id in camera_ids}
        self.video_writers = {}
        self.img_writers = {}
        self.cv_bridge = cvb.CvBridge()
        
        if not os.path.exists(os.path.join(self.output_dir, f"visual_observations")):
            os.makedirs(os.path.join(self.output_dir, f"visual_observations"))
        for i in self.camera_ids:
            i = i.replace("/", "")
            if not os.path.exists(os.path.join(self.output_dir, f"visual_observations", f"{i}")):
                os.makedirs(os.path.join(self.output_dir, f"visual_observations", f"{i}"))

        self.robot_data = {}
        self.data = {}
        self.indy = IndyDCP3('10.208.112.143')
        self.demo = 0
        self.key_listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.key_listener.start()
        self.quit = False

    def img_cb(self,data, camera_id):
        self.get_logger().info(f"Got image:{camera_id}")  
        self.imgs[camera_id] = self.cv_bridge.imgmsg_to_cv2(data)
        # self.get_logger().info(f"Img shape:{self.imgs[camera_id].shape}")
    
    def depth_cb(self,camera_id, data):
        self.depth_imgs[camera_id] = self.cv_bridge.imgmsg_to_cv2(data)
    
    def start_write(self, num): 
        self.data[num] = {}
        for camera_id in self.camera_ids:
            camera_id = camera_id.replace("/", "")
            if self.video:
                video_writer = cv2.VideoWriter(f"{self.output_dir}/video_{camera_id}_{num}.avi", cv2.VideoWriter_fourcc(*'XVID'), 15, (640, 480))
                self.video_writers[camera_id] = video_writer
                self.data[num][f"video_{camera_id}"] = f"{self.output_dir}/video_{camera_id}_{num}.avi"
        
        self.timer = self.create_timer(1/30, self.get_data, callback_group=ReentrantCallbackGroup())
        
        self.robot_data = {}
    
    def stop_write(self):
        self.timer.cancel()
        for camera_id in self.camera_ids:
            if self.video:
                self.video_writers[camera_id].release()
            else: 
                break
        self.demo += 1
        self.get_logger().info(f"Recorded data for demo: {self.demo -1 }, starting new demo: {self.demo}")

    def get_robot_state(self):
        self.get_logger().info(f"starting get_robot_state")
        pose = self.indy.get_control_data()['p'] 
        self.get_logger().info(f"Got pose data")
        ft_data = self.indy.get_ft_sensor_data()
        list = []
        for name,value in ft_data.items():
            list.append(value)
        ft_data2 = np.array(list)
        # ft_data = np.zeros(6)
        pose_np = np.array(pose)
        p = pose_np[:3]
        euler_ang = pose_np[3:]
        rot = Rot.from_euler("xyz", euler_ang)
        R = rot.as_matrix()
        homog_pose = np.eye(4)
        homog_pose[:3,:3] = R
        homog_pose[:3,3] = p
        
        return homog_pose, ft_data2

    def get_data(self):
        time = (self.get_clock().now().nanoseconds)//1000
        demo = copy.deepcopy(self.demo)
        self.data[demo][time] = {}
        for camera_id in self.camera_ids:
            camera_id = camera_id.replace("/", "")
            if self.video:
                self.video_writers[camera_id].write(self.imgs[camera_id])
                self.get_logger().info(f"Wrote frames:{self.imgs[camera_id].shape}")
            else: 
                cv2.imwrite(f"{self.output_dir}/visual_observations/{camera_id}/{demo}_{time}.png", self.imgs[camera_id.replace("/", "")])
                self.get_logger().info(f"Wrote image:{camera_id}")
                
                self.data[demo][time]['camera_id'] =  f"{self.output_dir}/visual_observations/{camera_id}/{demo}_{time}.png"
        pose,ft_data = self.get_robot_state()
        
        self.data[demo][time]['robot_data'] = {'pose': pose.tolist(), 'ft_data': ft_data.tolist()}
        self.get_logger().info("Got robot data")    
    
    def on_press(self, key):
        if key == KeyCode(char='t'):
            self.start_write(self.demo)
            
        if key == KeyCode(char='s'): 
            with open(f"{self.output_dir}/data.json", 'w') as f:
                json.dump(self.data, f)
            self.get_logger().info("Saved data")

        if key == KeyCode(char='r'):
            self.stop_write()
            
            
    def on_release(self, key):
        
        if key == KeyCode(char='q'):
            self.get_logger().info("Exiting")
            self.key_listener.stop()
            raise KeyboardInterrupt


def main(args=None):
    # executor = MultiThreadedExecutor()
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_ids', nargs='+')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    executor = MultiThreadedExecutor()
    data_logger = DataLogger(args.camera_ids, args.output_dir)
    executor.add_node(data_logger)
    executor.spin()
    rclpy.shutdown()
    return

if __name__ == "__main__":
    main()