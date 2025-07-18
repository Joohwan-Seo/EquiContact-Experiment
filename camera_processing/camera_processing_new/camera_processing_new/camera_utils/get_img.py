import rclpy 
import open3d as o3d
from rclpy.node import Node 
from sensor_msgs.msg import Image 
import cv2

import cv_bridge as cvb 

class ImageSub(Node): 

    def __init__(self, camera_id, image_path):
        super().__init__('imgsub') 
        self.camera_id = camera_id 
        self.image_path = image_path
        self.image_sub = self.create_subscription(Image, self.camera_id + "/color/image_raw", self.img_cb, 10)
        self.image = None
        self.img_bridge = cvb.CvBridge()
        self.i = 0
    
    def img_cb(self, data): 
        self.img_msg = data 
        self.image = self.img_bridge.imgmsg_to_cv2(data)
        #if keyboard.is_pressed("s"): 
        print(f"Saving to img path:{self.image_path}")
        self.save_img(self.image, self.image_path)
        
        #    self.i+=1
        
    def save_img(self, img, file_path): 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_path, img_rgb)
        self.get_logger().info(f"Saved image to {file_path}")

if __name__ == "__main__": 
    import argparse
    from pathlib import Path
    # Collect events until released

    parser = argparse.ArgumentParser()
    parser.add_argument("camera_id")
    parser.add_argument("pcd_path")
    args = parser.parse_args()
    rclpy.init()
    img_sub = ImageSub(args.camera_id, args.pcd_path)
    
    rclpy.spin(img_sub)
    img_sub.destroy_node()
    rclpy.shutdown()
