import cv2
import numpy as np

# Ouvrez un flux vidéo à partir de votre webcam
cap = cv2.VideoCapture(0)

# Configurez le modèle de détection de pose
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

# Créez le modèle de détection de pose et initialisez la couche de détection
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if __name__ == '__main__':
    while True:
        # Lisez une frame du flux vidéo
        ret, frame = cap.read()
        frameCopy = np.copy(frame)
        if not ret:
            break

        # Redimensionnez la frame et préparez-la pour le modèle de détection de pose
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth / frameHeight
        inHeight = 368
        inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        # Envoyez la frame au modèle de détection de pose et obtenez les points de pose
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        # Chaque point de pose est représenté par une paire x, y coordonnées dans la frame de sortie
        points = []
        for i in range(nPoints):
            # Extraire l'abscisse et l'ordonnée du point de pose
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)



            # Ignorez les points de pose qui ont une probabilité inférieure à un seuil
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            if prob > 0.1:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append(None)

                # Dessinez les lignes entre les points de pose
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                print(points[2])

                #if points[partA] and points[partB]:
                #    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                #    cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                # Affichez la frame avec les points de pose et les lignes dessinées
            cv2.imshow("Pose Estimation", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Libérez les ressources
        cap.release()
        cv2.destroyAllWindows()

