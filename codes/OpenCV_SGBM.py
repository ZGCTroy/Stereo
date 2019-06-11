import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    cn = 3
    num = 12
    block_size = 20
    min_disp = 0
    num_disp = num * 16

    stereo = cv2.StereoSGBM_create(
        numDisparities=num_disp,
        blockSize=block_size,
        speckleRange=1,
        speckleWindowSize=10,
        uniquenessRatio=5,
    )


    for i in range(1, 14):
        print(os.path.join('../images/rectified_left', 'left' + str(i).zfill(2) + '.jpg'))
        imgL = cv2.imread(os.path.join('../images/rectified_left', 'left' + str(i).zfill(2) + '.jpg'), 0)
        imgR = cv2.imread(os.path.join('../images/rectified_right', 'right' + str(i).zfill(2) + '.jpg'), 0)

        disparity = stereo.compute(imgL, imgR)
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8U)

        plt.subplot(2, 2, 1)
        plt.imshow(imgL, 'gray')
        plt.title('left image')
        plt.subplot(2, 2, 2)
        plt.imshow(imgR, 'gray')
        plt.title('right image')
        plt.subplot(2, 2, 3)
        plt.imshow(disparity)
        plt.show()


