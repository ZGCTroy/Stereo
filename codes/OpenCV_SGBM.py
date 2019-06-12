import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    cn = 3
    num = 8
    block_size = 21
    min_disp = 0
    num_disp = num * 16

    stereo = cv2.StereoSGBM_create(
        numDisparities=num_disp,
        blockSize=block_size,
        speckleRange=2,
        speckleWindowSize=100,
        uniquenessRatio=10,
    )


    for i in range(1, 2):
        imgL = cv2.imread(os.path.join('../images/rectified_left', 'rectified_left' + str(i).zfill(2) + '.jpg'), 0)
        imgR = cv2.imread(os.path.join('../images/rectified_right', 'rectified_right' + str(i).zfill(2) + '.jpg'), 0)

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
        plt.subplot(2, 2, 4)
        plt.imshow(disparity, 'gray')
        plt.show()

        cv2.imwrite('./disparity_left01.jpg',disparity)


