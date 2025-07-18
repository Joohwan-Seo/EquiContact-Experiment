# import rclpy 
import cv2 
import numpy as np
from camera_processing_new.camera_utils.get_img import ImageSub
import os

from calibrate import est_marker_pose

import scipy.spatial.transform as transform 

def create_gmtx (rvecs, tvecs): 
    g_matrices = []
    for i in range(len(rvecs)): 
        rot_mat,_ = cv2.Rodrigues(rvecs[i])
        gmtx = np.hstack((np.array(rot_mat), np.array(tvecs[i].reshape((3,1)))))
        gmtx = np.vstack((gmtx, np.array([0,0,0,1]).reshape((1, 4))))
        g_matrices.append(gmtx)
    return g_matrices

def calibrate_extrinsics(img_path, marker_poses, intrinsics, distortion, show_image = False): 
    img = cv2.imread(img_path)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    # (cv2.aruco_corners, cv2.aruco_ids, cv2.aruco_rejected) = aruco_detector.detectMarkers(camera_img_gray)
    (corners, ids, rejected) = aruco_detector.detectMarkers(img2)
    #print('corners:', corners)
    if len(corners) > 0: 
        ids = ids.flatten()

        rvecs, tvecs, trash = est_marker_pose(corners, 0.1, intrinsics, distortion)
        #print(f"This is rotation vector:{rvecs}, translation vectors: {tvecs}")
        g_matrices = create_gmtx(rvecs, tvecs)
        camera_poses = []
        axis_points = np.float32([
            [0, 0, 0],  # Origin
            [0.09, 0, 0],  # X-axis
            [0, 0.09, 0],  # Y-axis
            [0, 0, 0.09]   # Z-axis
        ])
        marker_size = 0.1
        marker_points = np.array([[0, marker_size, 0],
                              [marker_size,0, 0],
                              [marker_size, marker_size, 0],
                              [0, marker_size, 0]], dtype=np.float32)
        img_points, _ = cv2.projectPoints(axis_points, rvecs[0], tvecs[0], intrinsics, distortion)
        img_points = np.int32(img_points).reshape(-1, 2)

        # cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 2)  # X-axis in red
        # cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 2)  # Y-axis in green
        # cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 2)  # Z-axis in blue

        # Show the image with detected Aprilcv2.arucos

        if show_image:
            cv2.imshow('Detected ArUco Markers', cv2.aruco.drawDetectedMarkers(img2, corners, ids))
            cv2.drawFrameAxes(img, intrinsics, distortion, rvecs[0], tvecs[0], 0.1)
            cv2.imshow("Detected markers", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        reorient = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.save("CalibrationVal/marker_pose_02.npy",g_matrices[0])
        print('Marker poses:', marker_poses)
        for i in range(len(marker_poses)):
            print(i)
            camera_poses.append(reorient.dot(marker_poses[i].dot(np.linalg.inv(g_matrices[i]))))
            print(f"This is the camera extrinsics:\n {camera_poses[0]}")
            print(f"This is the predicted aruco pose: \n {g_matrices[i]}") 
        return camera_poses[0]
    return None

def main(cam_num, img_path, markers_in_inch, intrinsics, distortion):
    

    print(f'Calibration of Camera number{cam_num}')

    img_dir = img_path + '/' + 'cam' + str(cam_num)

    length = len(os.listdir(img_dir))

    extrinsics = []
    inch_to_m = 0.0254

    images_dir = sorted(os.listdir(img_dir))

    for i in range(len(images_dir)): 
        print(f"Opening image:{images_dir[i]}, index: {i+1}")

        marker_pose = np.array([[-1, 0,  0, markers_in_inch[i][0]*inch_to_m + 0.05],
                                [0,  1,  0, markers_in_inch[i][1]*inch_to_m - 0.05],
                                [0,  0, -1,                                      0],
                                [0,  0,  0,                                      1]])
        extrinsic = calibrate_extrinsics(img_dir + "/" + images_dir[i], [marker_pose], intrinsics, distortion)
        extrinsics.append(extrinsic)

    extrinsics_np = np.asarray(extrinsics)
    rotations = transform.Rotation.from_matrix(extrinsics_np[:,:3,:3])
    avg_rotations = rotations.mean().as_matrix()

    avg_translations = np.mean(extrinsics_np[:, :3, -1], axis=0)

    extrinsics_avg = np.eye(4)
    extrinsics_avg[:3,:3] = avg_rotations
    extrinsics_avg[:3,3] = avg_translations

    return extrinsics_avg, extrinsics_np


if __name__ == "__main__":
    import argparse
    # img_path = 'CalibrationImages'

    # cam_num = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--extrinsics_path")
    parser.add_argument("--cam_num", default = "1")
    parser.add_argument("--img_path", default = "CalibrationImages")
    parser.add_argument("--dir", default=False)

    args = parser.parse_args()

    cam_num = args.cam_num
    img_path = args.img_path

    #Obtained from ORBBEC Viewer 3920 x 2160
    intrinsics_011 = np.array([[2250.39, 0, 1918.12],[0, 2247.89, 1075.21],[0, 0, 1]])
    intrinsics_021 = np.array([[2247.39, 0, 1908.26],[0,2247.07,1080.88],[0 , 0, 1]])

    distortion_011 = np.array([0.0710723,-0.0975273, 0.000281252,0.000276891,0.0391244])
    distortion_021 = np.array([0.0705178,-0.0967127,-1.43643e-05,0.000139044,0.0386873]) 

    markers_in_inch = np.array([
        [14, 0],
        [16, 0],
        [18, 0],
        [20, 0],
        [22, 0],
        [24, 0],
        [14, -3],
        [16, -3],
        [18, -3],
        [18, 3],
        [21, 3],
        [24, 3],
        [22, 18],
        [18, 18],
        [26, 18],
        [24, -11],
        [21, -11],
        [18, -11],
        [18, 7],
        [22, 7],
        [22, 14],
        [19, 14],
        [25, 14]
    ], dtype=np.float64)

    if cam_num == "1":
        intrinsics = intrinsics_011
        distortion = distortion_011
    elif cam_num == "2":
        intrinsics = intrinsics_021
        distortion = distortion_021
    else:
        print("Not supported")

    extrinsics, extrinsics_all = main(cam_num, img_path, markers_in_inch, intrinsics, distortion)
    np.save(args.extrinsics_path, extrinsics)
    np.save("CalibrationVal/platform_02_all_matrices.npy", extrinsics_all)

    print('Expected extrinsics:', extrinsics)


    