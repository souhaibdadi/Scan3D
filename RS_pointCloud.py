from datetime import datetime

import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as op
import copy
import time
DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]

def find_device_that_supports_advanced_mode() :
    ctx = rs.context()
    ds5_dev = rs.device()
    devices  = ctx.query_devices();
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No D400 product line device that supports advanced mode was found")

"""advnc_mode = rs.rs400_advanced_mode(find_device_that_supports_advanced_mode())"""


# Get each control's current value
#print("Depth Control: \n", advnc_mode.get_depth_control())
#print("RSM: \n", advnc_mode.get_rsm())
#print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
#print("Color Control: \n", advnc_mode.get_color_control())
#print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
#print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
#print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
#print("HDAD: \n", advnc_mode.get_hdad())
#print("Color Correction: \n", advnc_mode.get_color_correction())
#print("Depth Table: \n", advnc_mode.get_depth_table())
#print("Auto Exposure Control: \n", advnc_mode.get_ae_control())
#print("Census: \n", advnc_mode.get_census())

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
pipe_profile = pipeline.start(config)
align = rs.align(rs.stream.color)
#hole_filling0 = rs.hole_filling_filter(0)



"""# modifier preset
depth_sensor = pipe_profile.get_device().first_depth_sensor()
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
#print('preset range:'+str(preset_range))_
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
    print("%02d: %s",(i,visulpreset))
    if visulpreset == "High Density":
        depth_sensor.set_option(rs.option.visual_preset, i)

pointcloud = op.geometry.PointCloud()
geometrie_added = False
##########################################################"""




def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    op.visualization.draw_geometries([source_temp, target_temp])

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


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = op.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        op.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            op.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            op.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], op.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result



def crop_cloud_with_bbox(cloud, center=[0, 0], length=1.5, width=10, height=10):
    bbox = op.geometry.AxisAlignedBoundingBox(min_bound=(center[0] - width, center[1] - height, -length),
                                               max_bound=(center[0] + width, center[1] + height, +length))
    return cloud.crop(bbox)



voxel_size = 0.01
i = 0
# result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

##########################################################

# Streaming loop
try:
    vis = op.visualization.Visualizer()
    vis.create_window("Tests")

    time.sleep(4)
    print("==========================  GO =========================")

    #while i<15:
    while True:
        dt0 = datetime.now()
        # Wait for the next set of frames from the camera
        frames_1 = pipeline.wait_for_frames()
        frames_1 = align.process(frames_1)
        #frames_1 = hole_filling0.process(frames_1).as_frameset()
        profile = frames_1.get_profile()

        # Align depth and color frame
        depth_1 = frames_1.get_depth_frame()
        color_1 = frames_1.get_color_frame()
        if not depth_1 or not color_1:  # or not depth_2 or not color_2:
            continue

        # Create RGBD
        color_raw = op.geometry.Image(np.array(color_1.get_data()))
        depth_raw = op.geometry.Image(np.array(depth_1.get_data()))
        rgbd_image = op.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

        # Get intrinsic
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        pinhole_camera_intrinsic = op.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx,intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        w, h = intrinsics.width, intrinsics.height


        # Create Point cloud from rgbd
        pcd_1 = op.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
        pcd_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


        # filtrer la distance ICI


        # Save point cloud
        op.io.write_point_cloud("pointClouds/fragment"+str(i)+".pcd", pcd_1)

        pointcloud.clear()
        pointcloud += crop_cloud_with_bbox(pcd_1)

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True

        vis.update_geometry(pointcloud)

        vis.poll_events()
        vis.update_renderer()
        process_time = datetime.now() - dt0
        i = i + 1
        #print("FPS: " + str(1 / process_time.total_seconds()), "w : ", w, "h : ",h)

finally:
    pipeline.stop()