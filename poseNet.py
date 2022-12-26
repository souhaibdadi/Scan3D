import cv2
import posenet
import tensorflow as tf

if __name__ == '__main__':
    # Ouvrez la webcam
    video_capture = cv2.VideoCapture(0)

    # Configurez le modèle de détection de pose
    model = 101

    # Créez une session TensorFlow
    sess = tf.compat.v1.Session()

    model = posenet.load_model(model,sess)

    # Définissez la résolution de l'image en entrée du modèle
    input_resolution = 512

    # Définissez le seuil de confiance à utiliser pour filtrer les poses détectées
    score_threshold = 0.5

    # Définissez le nombre de personnes que vous souhaitez détecter
    max_pose_detections = 1

    # Définissez le nombre de points clés à détecter par personne
    max_keypoint_detections = 10

    while True:
        # Capturez une frame de la webcam
        success, image = video_capture.read()

        # Si la capture a réussi, utilisez le modèle PoseNet pour estimer la pose
        if success:
            poses, _ = model.estimate_multiple_poses(image, input_resolution, score_threshold, max_pose_detections,
                                                     max_keypoint_detections)

            # Pour chaque pose détectée, dessinez les points clés sur l'image
            for pose in poses:
                posenet.draw_keypoints(image, pose.keypoints, min_part_confidence=score_threshold)

            # Affichez l'image avec les poses estimées
            cv2.imshow('Webcam', image)

        # Si vous appuyez sur 'q', quittez la boucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fermez la fenêtre de la webcam et libérez la capture vidéo
    video_capture.release()
    cv2.destroyAllWindows()
