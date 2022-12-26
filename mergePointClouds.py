from datetime import datetime

import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as op
import copy
import time
import concurrent.futures


import keyboard #load keyboard package


geometrie_added = False
##########################################################

def preprocess_point_cloud(pcd, voxel_size):
    """
        Pré-traite un nuage de points en réduisant sa résolution avec une réduction de voxel et en calculant
        les caractéristiques FPFH (Fast Point Feature Histogram).
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size*2)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        op.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = op.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        op.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
        Effectue une registration générale basée sur la correspondance de caractéristiques entre deux nuages de points.
    """
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = op.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False,
        distance_threshold,
        op.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            op.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            op.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], op.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result




def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
        Effectue une registration générale rapide basée sur la correspondance de caractéristiques entre deux nuages de points.
    """
    distance_threshold = voxel_size * 0.01
    #print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)

    result = op.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            op.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance = distance_threshold))
    result

def remove_outlier(pcd, nb_neighbors, std_ratio):
    """
        Supprime les points considérés comme des bruits d'un nuage de points en utilisant une méthode de suppression d'outliers statistiques.
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    return cl

# fonction qui prend en parametre un point cloud et qui retourne un mesh
def mesh(pcd):
    """
        Crée un maillage à partir d'un nuage de points en utilisant l'algorithme de maillage de Poisson.
    """
    print(":: Start meshing")
    mesh, doubleVercto  = op.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=15)
    print(":: Finish meshing")
    return mesh

def generateOctree(pcd):
    octree = op.geometry.Octree(max_depth=10)
    octree.convert_from_point_cloud(pcd)
    return octree


def generateVoxelGrid(pcd):
    print(":: Voxel grid")
    voxel_grid = op.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.01)
    return voxel_grid


def refine_registration(source, target, voxel_size,result_ransac):
    """
        Affine un résultat de registration en utilisant l'algorithme d'Iterative Closest Point (ICP) avec une correspondance de type point-to-plane.
    """
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = op.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        op.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def crop_cloud_with_bbox(cloud, center=[0, 0], length=1, width=5, height=5):
    """
       Coupe un nuage de points en utilisant une boîte englobante alignée sur les axes.
    """
    bbox = op.geometry.AxisAlignedBoundingBox(min_bound=(center[0] - width, center[1] - height, -length),
                                               max_bound=(center[0] + width, center[1] + height, +length))
    return cloud.crop(bbox)


def transformation_apply(pcd, transformation):
    """
        Applique une transformation à un nuage de points et calcule les caractéristiques FPFH (Fast Point Feature Histogram).
    """
    # Appliquer les transformations
    print(":: Start transformation")
    pcd_transformed = pcd.transform(transformation.transformation)

    radius_normal = voxel_size * 2
    pcd_transformed.estimate_normals(
        op.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = op.pipelines.registration.compute_fpfh_feature(
        pcd_transformed,
        op.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

    return pcd_transformed, pcd_fpfh



voxel_size = 0.01
i = 0


def mergePointClouds(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    pointcloud = op.geometry.PointCloud()

    target = op.io.read_point_cloud("pointClouds/fragment0.pcd")
    target = crop_cloud_with_bbox(target)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    pointcloud += target_down

    for i in range(5,50,3):
        dt0 = datetime.now()
        source = op.io.read_point_cloud("pointClouds/fragment"+str(i)+".pcd")
        source = crop_cloud_with_bbox(source)
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        #result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_ransac = refine_registration(source_down, target_down, voxel_size, result_ransac)

        process_time = datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))

        #Appliquer les transformations
        print(":: Start transformation")
        #source_down = source_down.transform(result_ransac.transformation)
        source_down, source_fpfh = transformation_apply(source_down,result_ransac)

        pointcloud += source_down
        print("Image ",str(i),"/98")
        target_down = source_down
        target_fpfh = source_fpfh
        #preprocess_point_cloud(pointcloud, voxel_size)


    pointcloud, _ = preprocess_point_cloud(pointcloud, voxel_size)
    pointcloud = remove_outlier(pointcloud, 40, 2)
    op.io.write_point_cloud("pointClouds/resultat1.pcd", pointcloud)

    vis = op.visualization.Visualizer()
    vis.create_window("Tests")
    vis.add_geometry(pointcloud)

    while True:
        vis.poll_events()
        vis.update_renderer()


def seePcds():

    vis = op.visualization.Visualizer()
    vis.create_window("Tests")

    source = op.io.read_point_cloud("pointClouds/resultat1.pcd")
    vis.add_geometry(source)

    while True:
        vis.poll_events()
        vis.update_renderer()


def mergeTwoPointClouds(voxel_size):
    pointcloud = op.geometry.PointCloud()

    source = op.io.read_point_cloud("pointClouds/fragment40.pcd")
    source = crop_cloud_with_bbox(source)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    target = op.io.read_point_cloud("pointClouds/fragment45.pcd")
    target = crop_cloud_with_bbox(target)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    print(result_ransac.transformation)

    #Applie transforation to the source
    source_down = source_down.transform(result_ransac.transformation)


    pointcloud += source_down.paint_uniform_color([1, 0.706, 0])
    pointcloud += target_down.paint_uniform_color([0, 0.651, 0.929])

    vis = op.visualization.Visualizer()
    vis.create_window("Tests")
    vis.add_geometry(pointcloud)

    while True:
        vis.poll_events()
        vis.update_renderer()

def createMeshFromPc():
    pointcloud = op.io.read_point_cloud("pointClouds/resultat1.pcd")
    body = generateOctree(pointcloud)
    vis = op.visualization.Visualizer()
    vis.create_window("Tests")
    vis.add_geometry(body)

    while True:
        vis.poll_events()
        vis.update_renderer()

if __name__ == '__main__':
    #mergePointClouds(voxel_size)
    #mergePointCloudsOpti(voxel_size)
    createMeshFromPc()
    #seePcds()





