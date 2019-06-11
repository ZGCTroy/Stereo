import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    camera = single_camera_system(image_root_dir='../images/right')

    camera.calibrate()

    for i in range(1, 2):
        img = cv2.imread(os.path.join('../images/left', 'left' + str(i).zfill(2) + '.jpg'))

        undistorted_img = camera.undistort(image=img)

        plt.figure(dpi=1200)
        plt.subplot(1, 2, 1)
        plt.title('origin image')
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(undistorted_img)
        plt.title('undistorted image')
        plt.show()

        # cv2.imwrite('./image_before_undistortion.jpg',img)
        # cv2.imwrite('./image_after_undistortion.jpg',undistorted_img)