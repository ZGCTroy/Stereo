import os
import numpy as np
from Camera import Camera
from StereoCamera import StereoCamera
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create a stereo camera system
    stereo_camera = StereoCamera(
        left_camera=Camera(image_root_dir='../images/left'),
        right_camera=Camera(image_root_dir='../images/right'),
    )

    # stereo calibrate
    stereo_camera.calibrate()

    # calculate the map function of the undistortion and rectify
    stereo_camera.get_rectify_map()

    print('calibration error:')
    print(stereo_camera.calibrate_error,'\n')

    print('left camera matrix:')
    print(stereo_camera.left_camera.camera_matrix,'\n')

    print('left camera \'s distortion coefficients:')
    print(stereo_camera.left_camera.distortion_coefficients,'\n')

    print('right camera matrix:')
    print(stereo_camera.right_camera.camera_matrix,'\n')

    print('right camera \'s distortion coefficients:')
    print(stereo_camera.right_camera.distortion_coefficients,'\n')

    print('rotation matrix R:')
    print(stereo_camera.R,'\n')

    print('translation vector T:')
    print(stereo_camera.T,'\n')

    print('essential matrix E:')
    print(stereo_camera.E,'\n')

    print('fundamental matrix F:')
    print(stereo_camera.F,'\n')