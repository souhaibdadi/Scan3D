from datetime import datetime

import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as op
import copy





def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        op.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = op.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        op.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def create_bbox(min_bound=(-3., -3., 0), max_bound=(3., 3., 2), flip=False):
    if flip:
        min_bound_ = min_bound.copy()
        max_bound_ = max_bound.copy()
        min_bound = [min_bound_[0], -max_bound_[1], -max_bound_[2]]
        max_bound = [max_bound_[0], -min_bound_[1], -min_bound_[2]]
    bbox = op.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox

def crop_cloud_with_bbox(cloud, center=[0, 0], length=1, width=5, height=5):
    bbox = op.geometry.AxisAlignedBoundingBox(min_bound=(center[0] - width, center[1] - height, -length),
                                               max_bound=(center[0] + width, center[1] + height, +length))
    return cloud.crop(bbox)


def showMeTheResult(pc,name):
    vis = op.visualization.Visualizer()
    vis.create_window(name)

    #inCloud = pc.select_by_index(ind).paint_uniform_color([1,0,0])
    #outCloud = pc.select_by_index(ind, invert=True).paint_uniform_color([0.8,0.8,0.8])
    #vis.add_geometry(inCloud)
    #vis.add_geometry(outCloud)

    axes = op.geometry.TriangleMesh.create_coordinate_frame()

    vis.add_geometry(crop_cloud_with_bbox(pc))
    vis.add_geometry(axes)


    while True:
        vis.poll_events()
        vis.update_renderer()



voxel_size = 0.02

def prepare_dataset(voxel_size):
    print("coucou")

    pointcloud = op.geometry.PointCloud()

    source = op.io.read_point_cloud("pointClouds/fragment9.pcd")

    #source_down, source_fpfh = preprocess_point_cloud(source,voxel_size)
    #print("------------------------------")
    #print(source.points)
    #print("------------------------------")
    #print(source_down)
    #print("------------------------------")
    #print(np.asarray(source_down.points))


    pointcloud += source

    #cl, ind = pointcloud.remove_radius_outlier(nb_points=20,radius=0.05)

    showMeTheResult(pointcloud,"voxel down")


prepare_dataset(voxel_size)