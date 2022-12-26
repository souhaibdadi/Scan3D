import cv2 as cv
import numpy as np
import pyrealsense2.pyrealsense2 as rs

if __name__ == '__main__':

    ctx = rs.context()
    print(len(ctx.devices))

    for i in range(len(ctx.devices)):
        sn = ctx.devices[i].get_info(rs.camera_info.serial_number)
        print(sn)

    cap = cv.VideoCapture(0)

    while True:
        sucess, img = cap.read()

        cv.imshow("Camera", img)


        if cv.waitKey(1) & 0xFF == ord('q'):
            break