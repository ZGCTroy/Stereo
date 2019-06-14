import os
import numpy as np
from Camera import Camera
from StereoCamera import StereoCamera
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    camera = Camera(image_root_dir='../images/right')

    camera.calibrate()

    image = cv2.imread('../images/right/right03.jpg')

    print('error, the total sum of squared distances between the observed feature points imagePoints and the projected object points objectPoints :')
    print(camera.error,'\n')

    print('camera matrix :')
    print(camera.camera_matrix,'\n')

    print('distortion coefficients')
    print(camera.distortion_coefficients,'\n')

