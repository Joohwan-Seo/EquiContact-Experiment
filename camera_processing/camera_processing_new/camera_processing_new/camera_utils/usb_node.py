import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class USBCam(Node): 

    def __init__(self, cam_id, cam_topic): 
        super().__init__('usb_cam_node')
        self.cam_id = cam_id
        self.cam_topic = cam_topic
        self.publisher = self.create_publisher(Image, cam_topic, 10)
        self.bridge = CvBridge()
        self.init_camera(cam_id)
        self.img_loop = self.create_timer(1/30, self.read_image)
    
    def init_camera(self, cam_id="/dev/v4l/by-id/usb-Arducam_Arducam_B0510__USB3_5MP_-video-index0"):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

        # Check if the camera opened successfully
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Set autofocus (1 for on, 0 for off)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(2)

        # Set focus value
        self.cap.set(cv2.CAP_PROP_FOCUS, 123)

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"{self.cam_id} camera initialized successfully")
        
    def read_image(self):
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Press 'q' to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        imgmsg = self.bridge.cv2_to_imgmsg(frame, "rgb8")
        self.publisher.publish(imgmsg)
        
if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='USB Camera Node')
    parser.add_argument('--cam_id', type=str, default="/dev/v4l/by-id/usb-Arducam_Arducam_B0510__USB3_5MP_-video-index0", help='Camera ID')
    parser.add_argument('--cam_topic', type=str, default='/arducam/color/image_raw', help='Camera topic name')
    args = parser.parse_args()
    rclpy.init()
    usb_cam = USBCam(args.cam_id, args.cam_topic)
    rclpy.spin(usb_cam)
    rclpy.shutdown()
