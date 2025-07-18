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

class ImageSub(Node):
    def __init__(self, camera_id, save_path, setpoints, method):
        super().__init__('imgsub')
        self.camera_id = camera_id
        self.save_path = save_path
        self.setpoints = setpoints
        self.method = method
        self.current_setpoint = 0  # Track the current setpoint index

        self.image_sub = self.create_subscription(Image, self.camera_id + "/color/image_raw", self.img_cb, 10)
        self.image = None
        self.img_bridge = cvb.CvBridge()
        self.indy = IndyDCP3("10.208.112.143")

        self.images = []
        self.robot_poses = []
        self.collected = 0  # Track collected data points

        self.intrinsics = np.array([[910.1775512695312, 0, 629.8541870117188],
                                    [0, 910.3551025390625, 384.5003356933594],
                                    [0, 0, 1]])
        self.distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])


        self.start_calibration()

    def start_calibration(self):
        self.get_logger().info("Starting hand-eye calibration...")
        self.move_to_next_setpoint()

    def move_to_next_setpoint(self):
        if self.current_setpoint < len(self.setpoints):
            target_position = self.setpoints[self.current_setpoint]
            self.get_logger().info(f"Moving to setpoint {self.current_setpoint + 1}: {target_position}")
            self.indy.movej(target_position)
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
            file_path = f"{self.save_path}/image_{self.collected}_charuco"

        cv2.imwrite(file_path + ".png", img_rgb)

        # Get robot pose
        p = self.indy.get_control_data()['p']
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
    parser.add_argument("--camera_id", default='/realsense/camera')
    parser.add_argument("--save_path", default='/home/horowitzlab/ros2_ws/realsense_calibration')
    parser.add_argument("--method", default = 'charuco')
    args = parser.parse_args()

    # Define setpoints (replace with actual robot positions)
    if args.method == "checker":
        setpoints = [
            [38.73734, 17.127186, -120.14716, -18.10487, -58.31407, 135.7586],
            [59.874172, 10.797881, -118.74899, -24.640213, -61.212936, 159.79097],
            [1.2652891, 8.734993, -105.34082, -0.65102154, -58.415504, 91.52917],
            [-11.8747225, -2.5462859, -75.82632, 20.747625, -87.505615, 34.815403],
            [5.502062, 4.7595315, -85.32573, 15.824936, -80.02588, 50.470486],
            [48.475357, -5.6140137, -79.453804, 0.00038071434, -94.942, 93.44225],
            [41.58189, -11.267978, -72.70361, 0.0018491839, -96.01458, 86.58684],
            [78.52696, 22.020597, -118.09853, -24.898882, -79.60454, 172.09676],
            [21.114634, 12.398889, -104.62353, -9.696304, -64.56752, 113.46478],
            [50.75248, 12.094176, -120.52324, -26.455948, -69.421196, 179.69695],
            [51.724403, 7.749311, -117.81389, -26.849878, -77.473465, 177.16411],
            [61.938873, 0.61577916, -110.50775, -24.924171, -81.81812, 185.85098],
            [49.26401, 16.226353, -123.83926, -26.330149, -60.501545, 182.26324], 
            [63.765057, 26.828568, -129.96295, -27.22499, -70.755, 193.6702],
            [36.906845, -13.374629, -67.14635, -9.235097, -95.47179, 155.72702],
            [33.004654, -12.137169, -64.15071, -0.52250326, -93.72665, 92.92753],
            [21.094477, -11.167693, -64.29131, 1.5380859, -94.64798, 81.39112],
            [17.97681, -40.010574, -46.25235, -2.1293898, -103.534065, 77.66687],
            [28.632095, -43.212452, -39.294056, -0.2546435, -107.498665, 88.55203],
            [10.070105, -40.5736, -40.59934, 6.6942635, -106.41118, 131.56482]
        ]
    elif args.method == "charuco":
        setpoints = [
            [42.137745, -0.10872833, -42.62173, -18.55705, -116.95468, 120.72169],
        ]

    rclpy.init()
    img_sub = ImageSub(args.camera_id, args.save_path, setpoints, method = args.method)
    rclpy.spin(img_sub)
    img_sub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
