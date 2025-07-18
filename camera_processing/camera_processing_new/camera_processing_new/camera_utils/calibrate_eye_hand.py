import rclpy 
import cv2 
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
from calibrate import est_marker_pose
from camera_processing_new.camera_utils.get_img import ImageSub

def calibrate_extrinsics(img_path, intrinsics, distortion): 
    img = cv2.imread(img_path)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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

def calibrate_img_dir(img_dir, intrinsics, distortion): 
    marker_R = []
    marker_t = []

    length = len(os.listdir(img_dir))

    for i in sorted(os.listdir(img_dir)): 
        print(f"Opening image:{i}")
        R, t = calibrate_extrinsics(img_dir + "/" + i, intrinsics, distortion)
        marker_R.append(R)
        marker_t.append(t)
    return marker_R, marker_t 

def cal_eye_hand(gripper_rot, gripper_pos, img_dir, intrinsics, distortion): 
    gripper_pos_arr= np.load(gripper_pos)
    gripper_rot_arr = np.load(gripper_rot)

    print(gripper_pos_arr)

    gripper_pos_inv_arr = gripper_pos_arr
    gripper_rot_inv_arr = gripper_rot_arr
    for i in range(gripper_pos_arr.shape[0]):
        gripper_pos_inv_arr[i] = (-gripper_rot_arr[i].T @ (gripper_pos_arr[i]).reshape((-1,1))).reshape((-1,))
        gripper_rot_inv_arr[i] = gripper_rot_arr[i].T
    
    marker_R, marker_t = calibrate_img_dir(img_dir, intrinsics, distortion)
    cam_rot, cam_t = cv2.calibrateHandEye(gripper_rot_inv_arr[:-2], gripper_pos_inv_arr[:-2], marker_R, marker_t)
    cam_gmtx = np.hstack((np.array(cam_rot), np.array(cam_t.reshape((3,1)))))
    cam_gmtx = np.vstack((cam_gmtx, np.array([0,0,0,1]).reshape((1, 4))))

    cam_gmtx = np.eye(4)
    cam_gmtx[:3,:3] = cam_rot
    cam_gmtx[:3,3] = cam_t
    print(cam_gmtx)
    np.save("CalibrationVal/handeye_02.npy", cam_gmtx)

if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper_rot")
    parser.add_argument("--gripper_pos")
    parser.add_argument("--img_dir")
    parser.add_argument("--intrinsics")
    parser.add_argument("--distortion")

    args = parser.parse_args()
    intrinsics = np.load(args.intrinsics)
    distortion = np.load(args.distortion)
    cal_eye_hand(args.gripper_rot, args.gripper_pos, args.img_dir, intrinsics, distortion)
