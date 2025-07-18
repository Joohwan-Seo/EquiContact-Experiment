import rclpy
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
import cv_bridge as cvb
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import argparse
from pathlib import Path
from neuromeka import IndyDCP3
from camera_processing_new.camera_utils.calibrate import est_marker_pose
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time

class ImageSubOrbbecMulti(Node):
    def __init__(self, save_path, number = '00'):
        super().__init__('imgsub')
        self.save_path = save_path
        self.current_setpoint = 0  # Track the current setpoint index

        self.number = number

        self.rgb_sub_1 = Subscriber(self, Image, 'camera_01/color/image_raw')
        self.rgb_sub_2 = Subscriber(self, Image, 'camera_02/color/image_raw')
        # self.rgb_camera_info_sub_1 = Subscriber(self, CameraInfo, 'camera_01/color/camera_info')
        # self.rgb_camera_info_sub_2 = Subscriber(self, CameraInfo, 'camera_02/depth/camera_info')

        self.image = None
        self.bridge = cvb.CvBridge()

        self.save = True

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub_1, self.rgb_sub_2],
            queue_size=10,
            slop=0.05,
        )
        self.ts.registerCallback(self.callback)

    def callback(self, rgb_msg_1, rgb_msg_2):
        try:
            # Convert ROS images to OpenCV images
            rgb_image_1 = self.bridge.imgmsg_to_cv2(rgb_msg_1, desired_encoding='bgr8')
            rgb_image_2 = self.bridge.imgmsg_to_cv2(rgb_msg_2, desired_encoding='bgr8')

            self.get_logger().info(f"Imgs Received")

            # Show undistorted RGB and depth
            cv2.imshow('Camera_01', rgb_image_1)
            cv2.imshow('Camera_02', rgb_image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Optionally, save the images from cv
            if self.save:
                cv2.imwrite(f"{self.save_path}/camera_01/image_{self.number}.png", rgb_image_1)
                cv2.imwrite(f"{self.save_path}/camera_02/image_{self.number}.png", rgb_image_2)
            else:
                pass

        except Exception as e:
            self.get_logger().error(f"Error processing images: {str(e)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default='/home/horowitzlab/ros2_ws/orbbec_calibration/multiview')
    parser.add_argument("--method", default = 'charuco')
    parser.add_argument("--number", default = '00')
    args = parser.parse_args()

    rclpy.init()
    img_sub = ImageSubOrbbecMulti(args.save_path, number=args.number)

    try:
        rclpy.spin(img_sub)
    except KeyboardInterrupt:
        pass
    finally:
        img_sub.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
