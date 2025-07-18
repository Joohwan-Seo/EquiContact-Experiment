import numpy as np 
from scipy.spatial.transform import Rotation as R
import scipy 

def cart2se3( p): 
    p = np.array(p)
    rot = R.from_euler('xyz', p[3:],degrees=True).as_matrix()
    homog_pose = np.eye(4)
    homog_pose[:3,:3] = rot
    homog_pose[:3,3] = p[:3]/1000
    return homog_pose

def se32cart( g): 
    p = g[:3,3] * 1000 
    rot = g[:3,:3]
    rot = R.from_matrix(rot)
    a = rot.as_euler('xyz', degrees=True)
    return np.concatenate((p, a), axis = 0)

def get_rel_twist( pose1, pose2): 
    g_rel = np.linalg.inv(pose1) @ pose2
    twist, _ = se3_logmap(g_rel, type="rotmat")
    return twist                 

def se3_logmap( pose, type="euler"): 
    """
    Convert a pose from the robot to a logmap representation
    """
    if type == "euler":
        pose = np.array(pose)
        p = pose[:3]
        rot = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
    else: 
        rot = pose[:3,:3]
        p = pose[:3,3]
        
    w, theta = so3_logmap(rot)
    if theta == 0: 
        twist = np.concatenate((w*theta, p), axis = 0)
        return twist, np.eye(3)
    w_hat = hat_map(w, type="ang")
    # V = np.eye(3) +((1 - np.cos(theta))/(theta**2)) * w_hat + ((theta - np.sin(theta))/(theta**3)) * (w_hat @ w_hat)
    V = np.eye(3) * (1/theta) - (0.5 * w_hat) + ((1/theta) - (0.5 * (1/np.tan(theta/2)))) * (w_hat @ w_hat)
    v = V @ p
    rot = R.from_matrix(rot)
    twist = np.concatenate((rot.as_rotvec(), v*theta), axis = 0)
    return twist, V        
        
def hat_map(w, type="ang"):         
    if type == "ang": 
        w_hat = np.array([[0, -w[2], w[1]],
                            [w[2], 0, -w[0]],
                            [-w[1], w[0], 0]])
    elif type == "twist":
        t = w[3:]
        w_hat = np.array([[0, -w[2], w[1], t[0]],
                            [w[2], 0, -w[0],t[1]],
                            [-w[1], w[0], 0, t[2]], 
                            [0, 0, 0, 0]])
    return w_hat

def vee_map(w):
    """
    Convert a skew-symmetric matrix to a vector
    """
    if w.shape == (3, 3):
        return np.array([w[2, 1], w[0, 2], w[1, 0]])
    elif w.shape == (4, 4):
        return np.array([w[2, 1], w[0, 2], w[1, 0], w[3, 0], w[3, 1], w[3, 2]])
    else:
        raise ValueError("Input must be a 3x3 or 4x4 matrix.")
        
def so3_logmap(rot): 
    """
    Convert a rotation from the robot to a logmap representation
            """
    if np.sum(np.eye(3) - rot) < 1e-3:
        return np.zeros(3), 0
    r = R.from_matrix(rot)
    test_theta = np.arccos(np.clip((np.trace(r.as_matrix()) - 1) / 2, -1, 1))
    test_omega  = (1/(2*np.sin(test_theta))) * np.array([rot[2, 1] - rot[1, 2],rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return test_omega, test_theta         

def se3_expmap( twist): 
    twist_hat = hat_map(twist, type="twist")
    g = scipy.linalg.expm(twist_hat)
    return g

def command_to_pose_data(command):
    """
    convert a command format to a pose_data format
    """
    output = np.zeros((6,))
    output[:3] = command[:3] / 1000 # mm to m
    output[3:] = R.from_euler('xyz', command[3:], degrees=True).as_rotvec()
    
    return output

def command_to_hom_matrix(command):
    """
    convert a command format to a homogenous matrix
    """
    output = np.eye(4)
    output[:3, 3] = command[:3] / 1000 # mm to m
    output[:3, :3] = R.from_euler('xyz', command[3:], degrees=True).as_matrix()
    
    return output

def hom_matrix_to_pose_data(matrix):
    """
    convert a homogenous matrix to a pose_data format
    """
    output = np.zeros((6,))
    output[:3] = matrix[:3, 3] # in meter
    output[3:] = R.from_matrix(matrix[:3, :3]).as_rotvec()
    
    return output

def pose_data_to_hom_matrix(pose_data):
    """
    convert a pose_data format to a homogenous matrix
    """
    
    output = np.eye(4)
    output[:3, 3] = pose_data[:3] # in meter
    output[:3, :3] = R.from_rotvec(pose_data[3:]).as_matrix()
    
    return output

def batch_pose_to_hom_matrices(poses):
    """
    poses: (N,6) array, each row = [tx,ty,tz,  ωx,ωy,ωz]
    returns: (N,4,4) array of homogeneous transforms
    """
    N = poses.shape[0]
    trans = poses[:, :3]             # (N,3)
    rotvecs = poses[:, 3:6]          # (N,3)

    # batch convert rotvecs → 3×3 rotation matrices
    R_mats = R.from_rotvec(rotvecs).as_matrix()  # (N,3,3)

    # build N copies of the 4×4 identity
    g_batch = np.tile(np.eye(4)[None, :, :], (N, 1, 1))  # (N,4,4)

    # insert translation and rotation
    g_batch[:, :3, 3] = trans
    g_batch[:, :3, :3] = R_mats

    return g_batch

def hom_matrix_to_command(mat):
    """
    convert a homogenous matrix to a command format
    """
    output = np.zeros((6,))
    output[:3] = mat[:3, 3] * 1000 # m to mm
    output[3:] = R.from_matrix(mat[:3, :3]).as_euler('xyz', degrees=True)
    
    return output

def batch_rot6d_pose_to_hom_matrices(rot6d_poses):
    """
    rot6d_poses: (N,9) array, each row = [position, rot6d]
    returns: (N,4,4) array of homogeneous transforms
    """
    N = rot6d_poses.shape[0]
    pos = rot6d_poses[:, :3]             # (N,3)
    rot6d = rot6d_poses[:, 3:]          # (N,6)
    # Convert 6D rotation to rotation matrix

    R_mats = np.array([rot6d_to_rotm(rot6d_i) for rot6d_i in rot6d])  # (N,3,3)
    # build N copies of the 4×4 identity
    g_batch = np.tile(np.eye(4)[None, :, :], (N, 1, 1))  # (N,4,4)
    # insert translation and rotation
    g_batch[:, :3, 3] = pos
    g_batch[:, :3, :3] = R_mats

    return g_batch


def rotm_to_rot6d(rotm):
    return rotm[:3, :2].T.flatten()

def rotvec_to_rot6d(rotvec):
    r = R.from_rotvec(rotvec).as_matrix()

    return r[:3, :2].T.flatten()

def rot6d_to_quat(rot6d):
    """Convert 6D rotation representation to quaternion.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    # print(f"x: {x}, y: {y}, z: {z}")
    quat = R.from_matrix(np.column_stack((x, y, z))).as_quat()
    
    return quat

def rot6d_to_rotm(rot6d):
    """Convert 6D rotation representation to rotation matrix.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    return np.column_stack((x, y, z))

def rot6d_to_rotvec(rot6d):
    """Convert 6D rotation representation to rotation vector.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    rotm = rot6d_to_rotm(rot6d)
    return R.from_matrix(rotm).as_rotvec()

def command_to_rot6d_pose_data(command):
    """
    Convert a command format to a rot6d pose data format.
    command: [tx, ty, tz, ωx, ωy, ωz]
    returns: [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    """
    output = np.zeros((9,))
    output[:3] = command[:3] / 1000  # mm to m
    rotm = R.from_euler('xyz', command[3:], degrees=True).as_matrix()
    output[3:] = rotm_to_rot6d(rotm)  # Convert rotation matrix to 6D representation
    
    return output