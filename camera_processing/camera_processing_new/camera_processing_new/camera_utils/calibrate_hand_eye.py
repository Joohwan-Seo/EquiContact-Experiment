import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from calibrate import est_marker_pose
import matplotlib.pyplot as plt
from camera_processing_new.camera_utils.get_img_client import ImgClientAsync
from scipy.spatial.transform import Rotation as Rot
from neuromeka import IndyDCP3
import numpy as np 
import cv2
import pickle
import os
from pynput.keyboard import Key, Listener, KeyCode
from argparse import ArgumentParser

class HandEye(Node): 

    def __init__(self, camera_id, save_dir, num_imgs, delay, calibrate=False):
        super().__init__("handeye")
        self.camera_id = camera_id
        self.save_dir = save_dir
        self.num_imgs = num_imgs
        self.delay = delay  
        self.calibrate = calibrate
        if not self.calibrate:
            self.client = ImgClientAsync()
            self.init_num = 0 
            self.indy = IndyDCP3("10.208.112.143")
            self.grasp = {}

    def get_pose(self): 
        pose_arr = self.indy.get_control_data()['p']
        self.get_logger().info(f"Current pose: {pose_arr}")
        self.grasp[int(self.init_num)] = pose_arr
    
    def calibrate_extrinsics(self,img_path, intrinsics, distortion): 
        img = cv2.imread(img_path)
        
        
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imshow(img2)
        plt.show()
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        # (cv2.aruco_corners, cv2.aruco_ids, cv2.aruco_rejected) = aruco_detector.detectMarkers(camera_img_gray)
        (corners, ids, rejected) = aruco_detector.detectMarkers(img2)
        print('corners:', corners)
        if len(corners) > 0: 
            ids = ids.flatten()

            rvecs, tvecs, trash = est_marker_pose(corners, 0.1, intrinsics, distortion)
            print(f"This is rotation vector:{rvecs}, translation vectors: {tvecs}")
            #g_matrices = create_gmtx(rvecs, tvecs)
            rmat,_ = cv2.Rodrigues(rvecs[0])
            return rmat, tvecs[0]
        return None

    def calibrate_img_dir(self,img_dir, intrinsics, distortion): 
        marker_R = []
        marker_t = []
        mark2cam = {}
        length = len(os.listdir(img_dir))
        ignore_images = ["img01.png", "img02.png", "img03.png", "img05.png"]
        for i in sorted(os.listdir(img_dir)): 
            try:
                R, t = self.calibrate_extrinsics(os.path.join(img_dir, i), intrinsics, distortion)
                marker_R.append(R)
                marker_t.append(t)
            except Exception as e:
                pass
        return marker_R, marker_t
    
    def indy2homog(self, pose): 
        print(pose)
        pose_np = np.array(pose)
        p = pose_np[:3]
        euler_ang = pose_np[3:]
        rot = Rot.from_euler("xyz", euler_ang)
        R = rot.as_matrix()
        homog_pose = np.eye(4)
        homog_pose[:3,:3] = R
        homog_pose[:3,3] = p
        return homog_pose
    
    def read_pose_dict(self, dict_path): 
        gripper_R = []
        gripper_t = []
        with open(dict_path, 'rb') as f:
            pose_dict = pickle.load(f)
        print(f"Pose dict:{pose_dict}")
        for v in pose_dict.values():
    
            homog_pose = self.indy2homog(v)
            R = homog_pose[:3,:3] 
            t= homog_pose[:3,3]
            gripper_R.append(R)
            gripper_t.append(t)
        return gripper_R, gripper_t
    
    def cal_hand_eye(self, data_dict, img_dir, intrinsics, distortion): 

        gripper_R, gripper_t = self.read_pose_dict(data_dict)        
        marker_R, marker_t = self.calibrate_img_dir(img_dir, intrinsics, distortion)
        cam_rot, cam_t = cv2.calibrateHandEye(gripper_R, gripper_t, marker_R, marker_t)
        cam_gmtx = np.hstack((np.array(cam_rot), np.array(cam_t.reshape((3,1)))))
        cam_gmtx = np.vstack((cam_gmtx, np.array([0,0,0,1]).reshape((1, 4))))

        cam_gmtx = np.eye(4)
        cam_gmtx[:3,:3] = cam_rot
        cam_gmtx[:3,3] = cam_t.reshape((3,))
        print(cam_gmtx)
        np.save("CalibrationVal/handeye_02.npy", cam_gmtx)


    def on_press(self, key, ): 

        if key == KeyCode(char='t'):
            future = self.client.send_request(self.camera_id, self.save_dir, int(self.num_imgs), float(self.delay), int(self.init_num))
            rclpy.spin_until_future_complete(self.client, future)
            response = future.result()
            if response.success: 
                self.client.get_logger().info("Getting images succeeded")
            self.get_pose()
            self.init_num += 1 
            print(f"Starting file number:{self.init_num}") 
               
    def on_release(self,key):
        if key == Key.esc:
            with open(os.path.join(self.save_dir, f"calib_poses.pkl"), 'wb') as f:
                pickle.dump(self.grasp, f)
            path = os.path.join(self.save_dir, f"calib_poses.pkl")
            self.get_logger().info(f"Saving poses to {path}")
            return False

def main(): 
    rclpy.init()

    parser = ArgumentParser()
    parser.add_argument("--camera_id", default = "/camera_01")
    parser.add_argument("--save_dir", default = "/home/horowitzlab/ros2_ws/test_imgs")
    parser.add_argument("--num_imgs", default=1)
    parser.add_argument("--delay", default = 0.2)
    parser.add_argument("--keyboard", default=False, action="store_true")
    parser.add_argument("--collect", default=False, action="store_true")
    parser.add_argument("--calibrate", default=True, action="store_true")
    args = parser.parse_args()
    
    handeye = HandEye(args.camera_id, args.save_dir, int(args.num_imgs), float(args.delay), args.calibrate)
    if args.collect:
        listener = Listener(on_press=lambda x: handeye.on_press(x), on_release=handeye.on_release)
        with listener as listener:
            listener.join()
        return
    if args.calibrate:
        handeye.cal_hand_eye(os.path.join(args.save_dir, "calib_poses.pkl"), os.path.join(args.save_dir, "rgb_imgs"), 
        np.load("/home/horowitzlab/ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/CalibrationVal/intrinsics/realsense.npy"), 
        np.load("/home/horowitzlab/ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/CalibrationVal/distortions/realsense.npy"))
if __name__ == "__main__":
    main()