import cv2 as cv
import numpy as np 
import glob 

 
def calibrate_intrinsics(img_dir, board_size):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_size = (int(board_size[0]), int(board_size[1]))
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((int(board_size[0])*int(board_size[1]),3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(img_dir + "/*.png")
    mtx = None 
    dist = None
    img_shape = 0
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board_size, None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            print(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img,board_size, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    print(cv.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)
    return mtx, dist      

if __name__ == "__main__": 
    import argparse 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir") 
    parser.add_argument("board_size_x")
    parser.add_argument("board_size_y")

    args = parser.parse_args()
    mtx, dist = calibrate_intrinsics(args.img_dir, (args.board_size_x, args.board_size_y))
    np.save("camera_matrix.npy", mtx)
    np.save("distortion_coeff.npy", dist)
    print(mtx, dist)