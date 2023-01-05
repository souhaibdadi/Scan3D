
import cv2
import mediapipe as mp
import numpy as np




if __name__ == '__main__':
    print("MediaPipe Python Test")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    maxDistance = None

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8, static_image_mode=False, model_complexity=2, enable_segmentation=False) as pose:
        while cap.isOpened():
            success, image = cap.read()

            # Detect poes and render the result
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # render detections
            window = mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                # Extract the pose landmarks
                pose_landmarks = results.pose_landmarks.landmark
                #pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
                RIGHT_SHOULDER = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                LEFT_SHOULDER = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                # calculate distance between shoulders
                #distance = np.sqrt((RIGHT_SHOULDER.x - LEFT_SHOULDER.x)**2 + (RIGHT_SHOULDER.y - LEFT_SHOULDER.y)**2 + (RIGHT_SHOULDER.z - LEFT_SHOULDER.z)**2)

                #print(distance)# coordonnées de l'épaule droite (x1, y1) et de l'épaule gauche (x2, y2)
                x1, y1 = RIGHT_SHOULDER.x, RIGHT_SHOULDER.y
                x2, y2 = LEFT_SHOULDER.x, LEFT_SHOULDER.y

                # création de deux vecteurs numpy à partir des coordonnées
                v1 = np.array([x1, y1])
                v2 = np.array([x2, y2])

                # calcul de la distance en utilisant la fonction linalg.norm de numpy
                #distance = np.linalg.norm(v1 - v2)

                # calcul de la distance en utilisant la formule de distance euclidienne
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if maxDistance is None:
                    maxDistance = distance

                print(distance * 100 / maxDistance)

            except:
                pass


            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()