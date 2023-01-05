import cv2
import mediapipe as mp
import numpy as np
import math as m
import pyrealsense2 as rs
import open3d as op

"""
Class permettant d'analyser le corps humain à partir d'un flux vidéo
"""

class BodyAnalazer:


    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.maxDistance = None
        self.mp_drawing = mp.solutions.drawing_utils

    """
    Methode permettant de lancer l'analyse du corps humain
    """
    def run(self):

        pointcloud = op.geometry.PointCloud()
        geometrie_added = False

        vis = op.visualization.Visualizer()
        vis.create_window("Tests")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        # Start streaming
        pipe_profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        # Wait for the next set of frames from the camera
        frames_1 = pipeline.wait_for_frames()
        frames_1 = align.process(frames_1)

        # frames_1 = hole_filling0.process(frames_1).as_frameset()
        profile = frames_1.get_profile()

        # Align depth and color frame
        depth_1 = frames_1.get_depth_frame()
        color_1 = frames_1.get_color_frame()

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

        pointcloud.clear()
        pointcloud += pcd_1

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True

        vis.update_geometry(pointcloud)

        vis.poll_events()
        vis.update_renderer()

        """
            while self.cap.isOpened() :
                success, image = self.cap.read()
    
                landmark = self.getBodyLandmarks(image)
                self.getShuldersDistance(landmark)
                #self.getShouldersAngle(landmark)
    
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        """


    """
    Méthode permettant de récupérer les point de repére du corps humain avec leur coordonnées
    """
    def getBodyLandmarks(self, image):

        # Detect poes and render the result
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # window = self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks is None:
            return None

        return results.pose_landmarks.landmark


    """
    Méthode permettant de récupérer la distance entre les épaules
    """
    def getShuldersDistance(self,landmark):
        if landmark is not None:
            # Extract the pose landmarks
            RIGHT_SHOULDER = landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            LEFT_SHOULDER = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # calculate distance between shoulders
            x1, y1 = RIGHT_SHOULDER.x, RIGHT_SHOULDER.y
            x2, y2 = LEFT_SHOULDER.x, LEFT_SHOULDER.y

            # calcul de la distance en utilisant la formule de distance euclidienne
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Considerant que le corps est de face, on considère que la distance maximale est la distance entre les épaules
            if self.maxDistance is None:
                self.maxDistance = round(distance,2)

            absDistance = abs(round(distance, 2) * 100 / self.maxDistance)
            print("% Distance : " + str(absDistance))

            return absDistance

    """
    methonde permettant de calculer l'angle entre la caméra et le corps humain
    """
    def getShouldersAngle(self, landmark):
        if landmark is not None:
            # Extract the pose landmarks
            RIGHT_SHOULDER = landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            LEFT_SHOULDER = landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # calculate angle between shoulders
            x1, y1 = RIGHT_SHOULDER.x, RIGHT_SHOULDER.y
            x2, y2 = LEFT_SHOULDER.x, LEFT_SHOULDER.y
            angle = self.getAngle(x1, y1, x2, y2)
            # print("Angle : " + str(angle))

            # Calcul du vecteur reliant les deux épaules
            shoulder_vector = (x1 - x2, y1 - y2)

            # Calcul de l'angle en radians entre le vecteur des épaules et l'horizontale
            angle_rad = m.atan2(shoulder_vector[1], shoulder_vector[0])

            # Conversion de l'angle en degrés
            angle_deg =  angle_rad / m.pi
            print("Angle : " + str(round(angle_deg,3)))


            return angle

    """
    Methode permettant de culculer l'angle entre deux points
    """
    def getAngle(self,x1, y1, x2, y2):
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / m.pi) * theta
        return degree


"""    def find_device_that_supports_advanced_mode():
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices();
        for dev in devices:
            if dev.supports(rs.camera_info.product_id) and str(
                    dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
                if dev.supports(rs.camera_info.name):
                    print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
                return dev
        raise Exception("No D400 product line device that supports advanced mode was found")
"""



if __name__ == '__main__':
    bodyAnalazer = BodyAnalazer()
    bodyAnalazer.run()