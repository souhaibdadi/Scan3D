import open3d as o3d
import cv2 as cv
import numpy as np
import pyrealsense2.pyrealsense2 as rs

if __name__ == '__main__':
     #o3d.t.io.RealSenseSensor.list_devices()

     # Configure depth and color streams
     pipeline = rs.pipeline()
     config = rs.config()

     pipeline_wrapper = rs.pipeline_wrapper(pipeline)
     pipeline_profile = config.resolve(pipeline_wrapper)
     device = pipeline_profile.get_device()

     found_rgb = False
     for s in device.sensors:
          if s.get_info(rs.camera_info.name) == 'RGB Camera':
               found_rgb = True
               break
     if not found_rgb:
          print("The demo requires Depth camera with Color sensor")
          exit(0)



     config.enable_stream(rs.stream.depth, rs.format.z16, 30)

     # Start streaming
     pipeline.start(config)

     # Get stream profile and camera intrinsics
     profile = pipeline.get_active_profile()
     depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
     depth_intrinsics = depth_profile.get_intrinsics()
     w, h = depth_intrinsics.width, depth_intrinsics.height

     print("width : ", w, "height : ", h)

     pc = rs.pointcloud()
     decimate = rs.decimation_filter()
     decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

     #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
     #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


     #while True:
     #     # Wait for a coherent pair of frames: depth and color
     #     frames = pipeline.wait_for_frames()
     #     depth_frame = frames.get_depth_frame()
     #     depth_frame = decimate.process(depth_frame)
     #     pc.calculate(depth_frame)


     #     depth_frame_open3d = o3d.geometry.Image(np.array(depth_frame.get_data()))

     #     #o3d.visualization.draw_geometries([depth_frame_open3d])

     #     #rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(depth_frame_open3d, depth_frame_open3d)

     frames = pipeline.wait_for_frames()
     depth_frame = frames.get_depth_frame()
     depth_frame = decimate.process(depth_frame)
     print(type(pc.calculate(depth_frame)))



     #depth_image = np.asanyarray(depth_frame.get_data())

     #print(depth_image)

     #pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_frame,
     #                                                      o3d.camera.PinholeCameraIntrinsic(
     #                                                           o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
     #                                                      , np.identity(4), depth_scale=1000.0, depth_trunc=1000.0)

     #print(np.asarray(pcd.points)[1, :])

     #o3d.visualization.draw_geometries(depth_image, zoom=1,front=[0,4257,-0,2125,-0,8795], lookat=[0,4257,-0,2125,-0,8795],up=[0,4257,-0,2125,-0,8795])

     #depth_frame_open3d = rs.Image(depth_frame)
     #color_frame_open3d = rs.Image(color_frame)
     #rgbd_image = rs.create_rgbd_image_from_color_and_depth(
     #     color_frame_open3d, depth_frame_open3d)
     #pcd = create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
     #rs.draw_geometries([pcd])





     #while True:










