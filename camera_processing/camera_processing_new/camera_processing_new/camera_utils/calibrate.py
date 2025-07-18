import rclpy 
import cv2 
import numpy as np
# from camera_processing_new.camera_utils.get_img import ImageSub

def calibrate_extrinsics(img_path,marker_poses, intrinsics, distortion): 
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
        cv2.imshow('Detected ArUco Markers', cv2.aruco.drawDetectedMarkers(img2, corners, ids))
        cv2.drawFrameAxes(img, intrinsics, distortion, rvecs[0], tvecs[0], 0.1)
        cv2.imshow("Detected markers", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        reorient = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.save("CalibrationVal/marker_pose_02.npy",g_matrices[0])
        for i in range(len(marker_poses)): 
            camera_poses.append(reorient.dot(marker_poses[i].dot(np.linalg.inv(g_matrices[i]))))
            print(f"This is the camera extrinsics:\n {camera_poses[0]}")
            print(f"This is the predicted aruco pose: \n {g_matrices[i]}") 
        return camera_poses[0]
    return None

def create_gmtx (rvecs, tvecs): 
    g_matrices = []
    for i in range(len(rvecs)): 
        rot_mat,_ = cv2.Rodrigues(rvecs[i])
        gmtx = np.hstack((np.array(rot_mat), np.array(tvecs[i].reshape((3,1)))))
        gmtx = np.vstack((gmtx, np.array([0,0,0,1]).reshape((1, 4))))
        g_matrices.append(gmtx)
    return g_matrices

def est_marker_pose(corners, marker_size, intrinsics, distortion ):
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        obj_points = np.array([
        [0, 0, 0],
        [marker_size, 0, 0],
        [marker_size, marker_size, 0],
        [0, marker_size, 0]
    ], dtype=np.float32)

    # Estimate pose of the AprilTag
        success, rvec, tvec = cv2.solvePnP(
        obj_points,
        c.astype(np.float32),
        intrinsics,
        distortion
        )
        trash.append(success)
        rvecs.append(rvec)
        tvecs.append(tvec)


    return rvecs, tvecs, trash

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("camera")
    parser.add_argument("img_path")
    parser.add_argument("--dir", default=False)
    parser.add_argument("extrinsics_path")
    args = parser.parse_args()
    intrinsics_02 = np.array([[750.131, 0, 639.372],[0,749.296,358.402],[0 , 0, 1]])
    intrinsics_021 = np.array([[2247.39, 0, 1908.26],[0, 2247.07,1080.88],[0 , 0, 1]])
    np.save("CalibrationVal/intrinsics_camera_022.npy", intrinsics_02)
    intrinsics_01 = np.array([[749.13, 0, 636.086],[0, 749.033, 360.295],[0, 0, 1]])
    intrinsics_011 = np.array([[2250.39, 0, 1918.12],[0, 2247.89, 1075.21],[0, 0, 1]])
    
    np.save("CalibrationVal/intrinsics_camera_0121.npy", intrinsics_01)
    # tangential_distortion = np.array([0.000061,0.000044])
    # radial_distortion = np.array([8.126028, 5.170630, 0.000061, 0.000044, 0.246261, 8.441916, 7.920322, 1.357227])
    distortion_02 = np.array([0.0705178,-0.0967127,-1.43643e-05,0.000139044,0.0386873])
    distortion_021 = np.array([0.0705178,-0.0967127,-1.43643e-05,0.000139044,0.0386873]) #Obtained from ORBBEC Viewer
    np.save("CalibrationVal/distortion_camera_022.npy", distortion_02)
    distortion_01 = np.array([0.0705178,-0.0967127,-1.43643e-05,0.000139044,0.0386873])
    distortion_011 = np.array([0.0710723,-0.0975273,0.000281252,0.000276891,0.0391244]) #Obtained from ORBBEC Viewer
    np.save("CalibrationVal/distortion_camera_012.npy", distortion_01)
    # radial_distortion = np.zeros((8,1))
    #radial_distortion = np.array([1.357227, 7.920322, 0.000061, 0.000044, 8.441916, 0.246261, 5.170630, 8.126028])
    marker_pose = [np.array([[-1, 0, 0, 0.4572 + 0.05],[0, 1, 0, -0.05 ],[0, 0, -1, 0],[0, 0, 0, 1]])]     
    if args.camera == "1":
        extrinsics = calibrate_extrinsics(args.img_path, marker_pose, intrinsics_011, distortion_011)
        np.save(args.extrinsics_path, extrinsics)
    elif args.camera == "2": 
        extrinsics = calibrate_extrinsics(args.img_path, marker_pose, intrinsics_01, distortion_01) #TBD
        np.save(args.extrinsics_path, extrinsics)


#Cam1
#Color Intrinsic: fx:2250.39 fy:2247.89 cx:1918.12 cy:1075.21 width:3840 height:2160
#k1:0.0710723 k2:-0.0975273 k3:0.0391244 k4:0 k5:0 k6:0 p1:0.000281252 p2:0.000276891

#Color Intrinsic: fx:750.131 fy:749.296 cx:639.372 cy:358.402 width:1280 height:720
#k1:0.0710723 k2:-0.0975273 k3:0.0391244 k4:0 k5:0 k6:0 p1:0.000281252 p2:0.000276891

#############################################

#Cam2
#Color Intrinsic : fx:2247.39 fy:2247.07 cx:1908.26 cy:1080.88 width:3840 height:2160
#k1:0.0705178 k2:-0.0967127 k3:0.0386873 k4:0 k5:0 k6:0 p1:-1.43643e-05 p2:0.000139044

#fx:749.13 fy:749.023 cx:636.086 cy:360.295 width:1280 height:720
#k1:0.0705178 k2:-0.0967127 k3:0.0386873 k4:0 k5:0 k6:0 p1:-1.43643e-05 p2:0.000139044