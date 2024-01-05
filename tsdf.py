import open3d as o3d
from trajectory_io import *
import numpy as np

if __name__ == "__main__":
    # (1) read trajectory from .log file
    camera_poses = read_trajectory("../../TestData/RGBD/odometry.log")
    # (2) TSDF volume integration
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(
            "../../TestData/RGBD/color/{:05d}.jpg".format(i))
        depth = o3d.io.read_image(
            "../../TestData/RGBD/depth/{:05d}.png".format(i))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(camera_poses[i].pose))
    # (3) Extract a mesh
    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
