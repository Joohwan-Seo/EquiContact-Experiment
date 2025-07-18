import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from calibrate import est_marker_pose
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

    def __init__(self, camera_id, save_dir, num_imgs, delay, collect=False): 
        super().__init__("handeye")
        self.camera_id = camera_id
        self.save_dir = save_dir
        self.num_imgs = num_imgs
        self.delay = delay  
        if collect:
            self.client = ImgClientAsync()
        self.init_num = 0 
        self.indy = IndyDCP3("10.208.112.143")
        self.grasp = {}

    def get_pose(self): 
        pose_arr = self.indy.get_control_data()['p']
        self.get_logger().info(f"Current pose: {pose_arr}")
        self.grasp[int(self.init_num)] = pose_arr
    def create_camera_matrix(self, intrinsics):
        print(intrinsics)
        fx = intrinsics[0]
        fy = intrinsics[1]
        cx = intrinsics[2]
        cy = intrinsics[3]
        camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        return camera_matrix
    def create_distortion_coeffs(self, distortion):
        k1 = distortion[0]
        k2 = distortion[1]
        p1 = distortion[2]
        p2 = distortion[3]
        k3 = distortion[4]
        dist_coeffs = np.array([[k1, k2, p1, p2, k3]])
        return dist_coeffs


    def calibrate_extrinsics(self,img_path, intrinsics, distortion): 
        print(f"Opening image:{img_path}")
        img = cv2.imread(img_path)
        camera_matrix = intrinsics
        dist_coeffs = self.create_distortion_coeffs(distortion)
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)    
        # cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)

        # Board specifics: squaresX x squaresY, square_length, marker_length
        # (All lengths are in some consistent unit, e.g., meters or centimeters)
        squaresX = 12
        squaresY = 9
        square_length = 0.06
        marker_length = 0.045
        
        # board = cv2.aruco.CharucoBoard_create(
        #     squaresX=squaresX,
        #     squaresY=squaresY,
        #     squareLength=square_length,
        #     markerLength=marker_length,
        #     dictionary=aruco_dict
        # )

        board = cv2.aruco.CharucoBoard(
            (squaresX, squaresY), square_length, marker_length, aruco_dict
        )
        corners, ids, rejected = cv2.aruco.detectMarkers(img2, aruco_dict)
        # (cv2.aruco_corners, cv2.aruco_ids, cv2.aruco_rejected) = aruco_detector.detectMarkers(camera_img_gray)
        if len(corners) > 0:
            # Refine detected markers
            corners, ids, rejected, recoveredIds = cv2.aruco.refineDetectedMarkers(
                img2,
                board,
                corners,
                ids,
                rejected
            )

        #-----------------------------------------------------------------------
        # 5) Interpolate ChArUco corners
        #-----------------------------------------------------------------------
        if len(corners) > 0 and ids is not None:
            # Interpolate the charuco corners
            resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=img2,
                board=board
            )

            #-------------------------------------------------------------------
            # 6) Estimate board pose (R, t) if enough corners are detected
            #-------------------------------------------------------------------
            if resp > 3:  # typically need at least 4 corners
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids,
                    board=board,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvec=None,
                    tvec=None
                )
                if success:
                    # rvec: rotation in Rodrigues form
                    # tvec: translation vector [X, Y, Z]

                    # 
                    #-------------------------------------------------------------------
                    # 7) Draw Axes and corners for visualization
                    #-------------------------------------------------------------------
                    cv2.aruco.drawDetectedMarkers(img2, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(img2, charuco_corners, charuco_ids)
                    axis_length = squaresY*square_length  # length of the coordinate axes to draw (same unit as board)
                    cv2.drawFrameAxes(img2, camera_matrix, dist_coeffs, rvec, tvec, axis_length)
                else:
                    print("Pose estimation failed. Not enough valid corners or some error.")

            else:
                print("Not enough Charuco corners for pose estimation. Found:", resp)
        else:
            print("No markers detected or 'ids' is None.")

        

        #-----------------------------------------------------------------------
        # 9) Pose of the Charuco Board with respect to the base frame
        #-----------------------------------------------------------------------
        
        

        return rvec, tvec


    def calibrate_img_dir(self,img_dir, intrinsics, distortion): 
        marker_R = []
        marker_t = []
        mark2cam = {}
        img_path = os.path.join(img_dir, "rgb_imgs")
        length = len(os.listdir(img_path))

        for i in sorted(os.listdir(img_path)): 
            print(f"Opening image:{i}")
            if i == "img00.png":
                continue
            R, t = self.calibrate_extrinsics(os.path.join(img_path, i), intrinsics, distortion)
            if R is None or t is None:
                print(f"Pose estimation failed for image: {i}")
                continue
            marker_R.append(R)
            marker_t.append(t)
        return marker_R, marker_t
    
    def indy2homog(self, pose): 
        pose_np = np.array(pose)
        p = pose_np[:3] * 0.001
        euler_ang = pose_np[3:]
        rot = Rot.from_euler("xyz", euler_ang, degrees=True)
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
        for k in pose_dict.keys():
            v = pose_dict[k]
            homog_pose = self.indy2homog(v)
            R = homog_pose[:3,:3] 
            t= homog_pose[:3,3]
            gripper_R.append(R)
            gripper_t.append(t)
        return gripper_R, gripper_t
    
    def cal_hand_eye(self, data_dict, img_dir, intrinsics, distortion): 
        gripper_R, gripper_t = self.read_pose_dict(data_dict)
        gripper_R.pop(0)
        gripper_t.pop(0)      
        marker_R, marker_t = self.calibrate_img_dir(img_dir, intrinsics, distortion)
        print(len(gripper_R), len(gripper_t))
        print(len(marker_R), len(marker_t))
        cam_rot, cam_t = cv2.calibrateHandEye(gripper_R, gripper_t, marker_R, marker_t)
        cam_gmtx = np.hstack((np.array(cam_rot), np.array(cam_t.reshape((3,1)))))
        cam_gmtx = np.vstack((cam_gmtx, np.array([0,0,0,1]).reshape((1, 4))))

        cam_gmtx = np.eye(4)
        cam_gmtx[:3,:3] = cam_rot
        cam_gmtx[:3,3] = cam_t.reshape((3))
        print(cam_gmtx)
        np.save("/home/horowitzlab/ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/CalibrationVal/handeye_02.npy", cam_gmtx)


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
    parser.add_argument("--save_dir", default = "/home/horowitzlab/ros2_ws_clean/calib_imgs2")
    parser.add_argument("--num_imgs", default=1)
    parser.add_argument("--delay", default = 0.2)
    parser.add_argument("--keyboard", default=False, action="store_true")
    parser.add_argument("--collect", default=False, action="store_true")
    parser.add_argument("--calibrate", default=False, action="store_true")
    args = parser.parse_args()
    
    handeye = HandEye(args.camera_id, args.save_dir, int(args.num_imgs), float(args.delay))
    if args.collect:
        listener = Listener(on_press=lambda x: handeye.on_press(x), on_release=handeye.on_release)
        with listener as listener:
            listener.join()
        return
    if args.calibrate:
        handeye.cal_hand_eye(os.path.join(args.save_dir, "calib_poses.pkl"), args.save_dir, np.load("/home/horowitzlab/ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/CalibrationVal/intrinsics/realsense.npy"), np.load("/home/horowitzlab/ros2_ws_clean/src/camera_processing/camera_processing_new/camera_processing_new/CalibrationVal/distortions/realsense.npy"))

if __name__ == "__main__":
    main()
