import open3d as o3d
import numpy as np
import copy
#from open3d.geometry import voxel_down_sample,estimate_normals

voxel_size = 0.005
max_correspondence_distance_coarse = voxel_size * 20
max_correspondence_distance_fine = voxel_size * 2


def load_point_clouds(pcd_path, voxel_size=0.005):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals()

    return pcd_down


def pairwise_registration(pcd1_down, pcd2_down):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd1_down, pcd2_down, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for pcd1_down_id in range(n_pcds):
        for pcd2_down_id in range(pcd1_down_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[pcd1_down_id], pcds[pcd2_down_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            # if pcd2_down_id == pcd1_down_id + 1:  # odometry case
            #     odometry = np.dot(transformation_icp, odometry)
            #     print("using odometry")
            #     pose_graph.nodes.append(
            #         o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            #     pose_graph.edges.append(
            #         o3d.pipelines.registration.PoseGraphEdge(pcd1_down_id,
            #                                        pcd2_down_id,
            #                                        transformation_icp,
            #                                        information_icp,
            #                                        uncertain=False))
            # else:  # loop closure case
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(pcd1_down_id,
                                                pcd2_down_id,
                                                transformation_icp,
                                                information_icp,
                                                uncertain=True))
    return pose_graph

def merge_pcd(pcds): 
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = o3d.geometry.voxel_down_sample(pcd_combined,
                                                       voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])

def rotmat_x(theta):
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    return R
def draw_registration_result(pcd1_down, pcd2_down, transformation):
    pcd1_down_temp = copy.deepcopy(pcd1_down)
    pcd2_down_temp = copy.deepcopy(pcd2_down)
    # pcd1_down_temp.paint_uniform_color([1, 0.706, 0])
    # pcd2_down_temp.paint_uniform_color([0, 0.651, 0.929])
    pcd1_down_temp.transform(transformation)
    o3d.visualization.draw_geometries([pcd1_down_temp, pcd2_down_temp])

def color_icp(pcd1, pcd2):
    pcd1_copy = copy.deepcopy(pcd1)
    pcd2_copy = copy.deepcopy(pcd2)
    voxel_radius = [0.04 * 1000, 0.02 * 1000, 0.01 * 1000]
    max_iter = [50, 30, 14]

    # voxel_radius = [0.04 * 1000, 0.02 * 1000]
    # max_iter = [50, 30]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(len(voxel_radius) -1 ):
        print("SCALE!!!!!!!!!!!!", scale)
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = pcd1_copy.voxel_down_sample(radius)
        target_down = pcd2_copy.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp.transformation)
    return current_transformation


if __name__ == "__main__":
    import argparse 
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_1_path")
    parser.add_argument("--pcd_2_path")
    parser.add_argument("--extrinsics_1_path")
    parser.add_argument("--extrinsics_2_path")
    
    args = parser.parse_args()

    # marker_1_pos = np.load(args.extrinsics_1_path)
    # marker_2_pos = np.load(args.extrinsics_2_path)
    # relative_transform = marker_1_pos.dot(np.linalg.inv(marker_2_pos))

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd1 = Path(args.pcd_1_path)
    
    pcd1_down = load_point_clouds(args.pcd_1_path, voxel_size=1)

    

    #if not pcd1.exists(): 
    #    print("Point cloud 1 file does not exist")
    #else: 
    #    pcd1_down = load_point_clouds(pcd1, voxel_size=voxel_size)
    
    pcd2 = Path(args.pcd_2_path)
    pcd2_down = load_point_clouds(args.pcd_2_path, voxel_size=1)
    #if not pcd2.exists(): 
    #    print("Point cloud 1 file does not exist")
    #else: 
    #    pcd2_down = load_point_clouds(pcd2, voxel_size=voxel_size)
    pcds_down = [pcd1_down, pcd2_down]
    extrinsics_01 = np.load(args.extrinsics_1_path)
    extrinsics_02 = np.load(args.extrinsics_2_path)

    extrinsics_01[:3,3] = extrinsics_01[:3,3]
    extrinsics_02[:3,3] = extrinsics_02[:3,3]

    print("extrinsics1:", extrinsics_01)
    print("extrinsics2:", extrinsics_02)
    
    pcds_down[0].transform(extrinsics_01)
    pcds_down[1].transform(extrinsics_02)

    offset = np.array([1, -0.5, 0.7])
    # offset_mat = np.array([[1, 0, 0, 0],[0, 1, 0, -0.015],[0, 0, 1, 0.015],[0, 0, 0, 1]])
    offset_mat = np.eye(4)
    theta_offset = 0 * np.pi
    R = rotmat_x(theta_offset)
    offset_mat[:3,:3] = R
    offset_mat[:3,3] = offset
    #np.save("CalibrationVal/new_camera_02.npy", offset_mat.dot(extrinsics_02))
    #pcd2_down = pcd2_down.transform(offset_mat)

    # min_bound = np.array([-0.8, -0.5, -0.1])
    # max_bound = np.array([-0.2, 0.3, 0.3])

    min_bound = np.array([0.2, -0.5, 0.0]) #* 1000
    max_bound = np.array([0.8, 0.6, 0.3]) #* 1000

    # min_bound = np.array([0.2, -0.5, 0.005])
    # max_bound = np.array([0.8, 0.6, 0.3])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd1_down = pcd1_down.crop(bbox)
    pcd2_down = pcd2_down.crop(bbox)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    
    cl, ind = pcd1_down.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.0)
    pcd1_down = pcd1_down.select_by_index(ind)
    pcd1_down = pcd1_down.voxel_down_sample(voxel_size=0.005)

    o3d.visualization.draw_geometries([pcd1_down, mesh_frame])

    
    cl, ind = pcd2_down.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.0)
    pcd2_down = pcd2_down.select_by_index(ind)
    pcd2_down = pcd2_down.voxel_down_sample(voxel_size=0.005)

    o3d.visualization.draw_geometries([pcd2_down, mesh_frame])

    pcds_down = [pcd1_down, pcd2_down]

    # pcds_down = [pcds_down[0].transform(extrinsics_01),pcds_down[1].transform(extrinsics_02)]
    #o3d.visualization.draw_geometries([pcds_down[0], mesh_frame])
    #o3d.visualization.draw_geometries([pcds_down[1], mesh_frame])
    
    # print("Full registration ...")
    # pose_graph = full_registration(pcds_down,
    #                                max_correspondence_distance_coarse,
    #                                max_correspondence_distance_fine)

    # print("Optimizing PoseGraph ...")
    # option = o3d.pipelines.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0)
    # o3d.pipelines.registration.global_optimization(
    #     pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

    # print("Transform points and display")
    # for point_id in range(len(pcds_down)):
    #     print(pose_graph.nodes[point_id].pose)
    #     pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    # #o3d.visualization.draw_geometries(pcds_down, mesh_frame)
    threshold = 20
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.eye(4)
    #draw_registration_result(pcd1_down, pcd2_down, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd1_down, pcd2_down,
                                                        threshold, trans_init)
    print(evaluation)

    ############################### This is the ICP we should use
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    # draw_registration_result(pcd1_down, pcd2_down, reg_p2p.transformation) 
    #pcd1_down.transform(reg_p2p.transformation)

    #######################################

    #######################################
    # transform = color_icp(pcd1_down, pcd2_down)
    pcd1_down.transform(reg_p2p.transformation)

    pcd_combined = pcd1_down + pcd2_down
    cl, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=1.0)
    pcd_combined= pcd_combined.select_by_index(ind)
    pcd_combined= pcd_combined.voxel_down_sample(voxel_size=1)

    new_extrinsic_01 = reg_p2p.transformation.dot(extrinsics_01)
    new_extrinsic_01[:3,3] = new_extrinsic_01[:3,3] * 0.001
    # np.save("CalibrationVal/new_camera_01.npy", new_extrinsic_01)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_combined,
                                                          voxel_size=0.002)
    o3d.visualization.draw_geometries([pcd_combined, mesh_frame])
    