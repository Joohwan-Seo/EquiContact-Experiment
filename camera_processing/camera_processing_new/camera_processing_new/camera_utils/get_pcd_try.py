import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import argparse
import struct


class ColoredPointCloudSubscriber(Node):
    def __init__(self, camera_id, save = True):
        super().__init__('colored_point_cloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            camera_id + '/depth_registered/points',  # Replace with your actual topic name
            self.point_cloud_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

        self.save = save
        self.camera_id = camera_id

    def point_cloud_callback(self, msg):
        self.get_logger().info("Received colored point cloud data")

        # Convert PointCloud2 to a numpy array
        points = []
        colors = []
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True):
            x, y, z, rgb = point
            points.append([x, y, z])

            if isinstance(rgb, (float, np.float32)):  # If rgb is float32, unpack to uint32
                packed = struct.unpack('I', struct.pack('f', rgb))[0]
            elif isinstance(rgb, int):  # If rgb is already uint32
                packed = rgb
            else:
                self.get_logger().warning(f"Unexpected RGB data type: {type(rgb)}")
                continue
            # Unpack RGB color
            r = (packed >> 16) & 0xFF
            g = (packed >> 8) & 0xFF
            b = packed & 0xFF
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize to [0, 1]

        self.get_logger().info(f"Processed {len(points)} points")
        # Convert to numpy arrays
        points_np = np.array(points, dtype=np.float32)
        colors_np = np.array(colors, dtype=np.float32)

        # Create Open3D point cloud
        if points_np.size > 0:
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(points_np)
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors_np)

            # Save the point cloud to a file
            if self.camera_id == "/camera_01":
                save_path = "/home/horowitzlab/test_pcd_01.pcd"
            elif self.camera_id == "/camera_02":
                save_path = "/home/horowitzlab/test_pcd_02.pcd"
            o3d.io.write_point_cloud(save_path, o3d_pcd)
            self.get_logger().info(f"Colored point cloud saved as {save_path}")

            # Optional: Visualize the point cloud
            o3d.visualization.draw_geometries([o3d_pcd])
        else:
            self.get_logger().warning("Empty point cloud received")


def main():
    rclpy.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=str, default="/camera_01")
    args = parser.parse_args()
    node = ColoredPointCloudSubscriber(args.camera_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
