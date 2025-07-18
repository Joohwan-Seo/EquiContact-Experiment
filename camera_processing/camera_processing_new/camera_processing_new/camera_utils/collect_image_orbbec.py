import rclpy
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import cv_bridge as cvb
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import argparse
from pathlib import Path
from neuromeka import IndyDCP3
from camera_processing_new.camera_utils.calibrate import est_marker_pose
import time

class ImageSubOrbbec(Node):
    def __init__(self, camera_id, save_path, setpoints, method, number = '01'):
        super().__init__('imgsub')
        self.camera_id = camera_id
        self.save_path = save_path
        self.setpoints = setpoints
        self.method = method
        self.current_setpoint = 0  # Track the current setpoint index

        self.number = number

        self.image_sub = self.create_subscription(Image, self.camera_id + "/color/image_raw", self.img_cb, 10)
        self.image = None
        self.img_bridge = cvb.CvBridge()
        self.indy = IndyDCP3("10.208.112.143")

        self.images = []
        self.robot_poses = []
        self.collected = 0  # Track collected data points


        self.start_calibration()

    def start_calibration(self):
        self.get_logger().info("Collecting calibration data...")
        self.move_to_next_setpoint()

    def move_to_next_setpoint(self):
        if self.current_setpoint < len(self.setpoints):
            target_position = self.setpoints[self.current_setpoint]
            self.get_logger().info(f"Moving to setpoint {self.current_setpoint + 1}: {target_position}")
            # self.indy.movej(target_position)
            time.sleep(5)  # Wait for the robot to reach the position (adjust as necessary)
            self.current_setpoint += 1
        else:
            self.get_logger().info("Collected all data points.")
            self.destroy_node()
            rclpy.shutdown()

    def img_cb(self, data):
        if self.current_setpoint <= len(self.setpoints):
            self.image = self.img_bridge.imgmsg_to_cv2(data)
            print(f"Saving image and pose {self.collected + 1}...")
            self.collect_data(self.image)
            self.collected += 1
            self.move_to_next_setpoint()

    def collect_data(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.method == "checker":
            file_path = f"{self.save_path}/image_{self.collected}"
        elif self.method == "charuco":
            file_path = f"{self.save_path}/image_charuco_{self.number}"
            self.get_logger().info(file_path)
        cv2.imwrite(file_path + ".png", img_rgb)

        # Get robot pose
        # p = self.indy.get_control_data()['p']
        p = np.zeros((6,))
        rotm = Rot.from_euler('xyz', p[3:6], degrees=True).as_matrix()
        pos = np.array(p[0:3]) * 0.001  # Convert mm to meters

        # save robot pose
        pose = np.eye(4)
        pose[:3, :3] = rotm
        pose[:3, 3] = pos

        if self.method == "checker":
            np.save(f"{self.save_path}/pose_{self.collected}", pose)
        elif self.method == "charuco":
            np.save(f"{self.save_path}/pose_{self.collected}_charuco", pose)
            

        # Append image and pose data
        self.images.append(img_rgb)
        self.robot_poses.append((pos, rotm))

    # def perform_hand_eye_calibration(self):
    #     robot_rotations = []
    #     robot_translations = []
    #     camera_rotations = []
    #     camera_translations = []

    #     for pose, image in zip(self.robot_poses, self.images):
    #         pos, rotm = pose

    #         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #         aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    #         parameters = cv2.aruco.DetectorParameters()
    #         aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    #         corners, ids, _ = aruco_detector.detectMarkers(gray)

    #         if len(corners) > 0: 
    #             ids = ids.flatten()

    #             rvecs, tvecs, trash = est_marker_pose(corners, 0.1, self.intrinsics, self.distortion)
    #             rmat,_ = cv2.Rodrigues(rvecs[0])
    #             camera_rotations.append(rmat)
    #             camera_translations.append(tvecs[0])

    #             robot_rotations.append(rotm)
    #             robot_translations.append(pos)

    #         else:
    #             self.get_logger().warn("No ArUco markers detected in image. Skipping...")


    #     # Solve hand-eye calibration
    #     r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    #         robot_rotations, robot_translations, camera_rotations, camera_translations, method=cv2.CALIB_HAND_EYE_TSAI
    #     )

    #     self.get_logger().info(f"Hand-eye calibration completed.")
    #     self.get_logger().info(f"Rotation:\n{r_cam2gripper}")
    #     self.get_logger().info(f"Translation:\n{t_cam2gripper}")

    #     # Save calibration result
    #     np.savez(f"{self.save_path}/hand_eye_calibration.npz", rotation=r_cam2gripper, translation=t_cam2gripper)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", default='/camera_01')
    parser.add_argument("--save_path", default='/home/horowitzlab/ros2_ws/orbbec_calibration')
    parser.add_argument("--method", default = 'charuco')
    parser.add_argument("--number", default = '00')
    args = parser.parse_args()

    args.save_path = args.save_path + "/" + args.camera_id

    # Define setpoints (replace with actual robot positions)
    if args.camera_id == '/camera_01':
        if args.method == "checker":
            setpoints = [
            [-18.427431, -6.813748, -99.151024, -25.07586, -85.14263, 26.179277],
            [-18.765602, 6.828911, -124.988686, -46.135616, -54.183266, 54.101685],
            [-21.753157, -21.20829, -81.65402, -18.367836, -112.306435, 9.288071],
            [-21.365276, -14.112165, -86.75123, -17.713442, -99.59922, 48.833466],
            [-18.267221, 0.3145631, -108.27049, -28.499134, -70.31718, 1.1000469],
            [-29.873323, -8.740124, -80.39895, 0.0033176534, -90.85547, 7.140787],
            [-13.693823, -8.313291, -121.05826, -52.80225, -87.00388, 44.68183],
            [-18.048403, -13.432965, -128.01787, -39.56661, -58.38336, 46.186523],
            [-16.181181, -6.0593233, -93.42401, -32.50033, -93.78854, 22.483465],
            [-6.5312815, 7.7159433, -118.116554, -29.629366, -89.25739, 36.103577],
            [-25.163912, -31.887272, -70.338425, -34.21311, -86.680336, 16.636782],
            [-20.74541, -17.815693, -96.85016, -43.885487, -65.15127, 41.746307],
            [-13.006496, -22.642641, -82.763824, -25.16946, -105.519165, 17.63447],
            [-9.195239, -27.125517, -110.024216, -39.294724, -89.01852, 35.286728],
            [-15.077056, -12.471389, -137.71866, -73.612366, -61.359783, 31.569431],
            [-12.874206, -35.062637, -91.49605, -25.441454, -105.5512, 41.859814],
            [-15.576888, -50.578968, -63.965443, -29.641983, -115.17854, 33.75397],
            [-11.27901, -9.272552, -96.573364, 11.459501, -91.33152, 46.38172],
            [-8.028055, -6.0618653, -88.86564, -3.0830247, -88.99943, 52.11011],
            [-5.3281875, -26.57057, -82.715836, -25.461794, -108.13989, 54.97809],
            ]
        #CHARUCO
        elif args.method == "charuco":
            setpoints = [
                [150, 0, -90, 0, -90, 0],
            ]
    elif args.camera_id == '/camera_02':
        if args.method == "checker":
            setpoints = [
                [4.60, -14, -70, 5, -103, 183],
                [11.43, -15, -63.6, 37.6, -93.6, 170.3],
                [9.86, -22, -63, 21, -41, 170],
                [16, -28, -65, 23, -50, 172.6],
                [5, -23, -84, 27, -45, 180],
                [16.6, -21.5, -56.6, 41, -82, 172.6],
                [4.6, -14, -73, -0.3, -74.2, 183],
                [12.3, -22.06, -64.38, 46.32, -62.5, 160.3],
                [4.6, -14.35, -65.75, 25.23, -111, 200],
                [6.9, -26, -75, 14, -66, 170.3],
                [4.5, -30, -82, 38, -51.7, 190],
                [11.4, -17.5, -67.97, 40, -96, 170],
                [9, -12, -68, 32, -78, 210],
                [8, -28.5, -65, 39.4, -33.7, 180],
                [9.3, -9.3, -62.8, 18.5, -91, 180],
                [9.3, -20.9, -70.3, 15.31, -72.52, 170],
                [9.3, -6.44, -62.79, 29.95, -107.53, 183],
                [9.3, -6.28, -62.77, 25.23, -108.18, 214],
                [9.3, -14.87, -62.83, 27.78, -77.88, 180.17],
                [10.24, -34.6, -70.70, 19.19, -26.88, 214],
            ]
        elif args.method == "charuco":
            setpoints = [
                [150, 0, -90, 0, -90, 0],
            ]

    rclpy.init()
    img_sub = ImageSubOrbbec(args.camera_id, args.save_path, setpoints, method=args.method, number=args.number)
    rclpy.spin(img_sub)
    img_sub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
