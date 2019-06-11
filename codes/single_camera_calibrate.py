import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    camera = single_camera_system(image_root_dir='../images/right')

    camera.calibrate()

    image = cv2.imread('../images/right/right03.jpg')

    undistorted_image = camera.undistort(image=image)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(undistorted_image)
    plt.show()