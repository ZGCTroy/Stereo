import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create a stereo camera system
    stereo_camera = stereo_camera_system(
        left_camera=single_camera_system(image_root_dir='../images/left'),
        right_camera=single_camera_system(image_root_dir='../images/right'),
    )

    # stereo calibrate
    stereo_camera.stereo_calibrate()

    # calculate the map function of the undistortion and rectify
    stereo_camera.calculate_undistort_and_rectify_map()

    print('left camera matrix')
    print(stereo_camera.left_camera.camera_matrix,'\n')

    print('left camera \'s distortion coefficients ')
    print(stereo_camera.left_camera.distortion_coefficients,'\n')

    print('right camera matrix')
    print(stereo_camera.right_camera.camera_matrix,'\n')

    print('right camera \'s distortion coefficients ')
    print(stereo_camera.right_camera.distortion_coefficients,'\n')

    print('rotation matrix R')
    print(stereo_camera.R,'\n')

    print('translation vector T')
    print(stereo_camera.T,'\n')

    print('essential matrix E')
    print(stereo_camera.E,'\n')

    print('fundamental matrix F')
    print(stereo_camera.F,'\n')