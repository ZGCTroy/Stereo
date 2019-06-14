import os
import numpy as np
from Camera import Camera
from StereoCamera import StereoCamera
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    camera = Camera(image_root_dir='../images/left')

    camera.calibrate()

    for i in range(1, 14):
        img = cv2.imread(os.path.join('../images/left', 'left' + str(i).zfill(2) + '.jpg'))

        undistorted_img = camera.undistort(image=img)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('origin image')
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(undistorted_img)
        plt.title('undistorted image')
        plt.show()