import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

from Camera import Camer

class StereoCamera():
    def __init__(self,
                 left_camera = None,
                 right_camera = None,
                 image_size = (640, 480),
                 R = None,
                 T = None,
                 E = None,
                 F = None
                 ):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.image_size = image_size
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.left_maps = None
        self.right_maps = None
        self.P1 = None
        self.P2 = None
        self.R1 = None
        self.R2 = None
        self.Q = None


    def calibrate(self):

        self.left_camera.calibrate()
        self.right_camera.calibrate()
        
        retval, self.left_camera.camera_matrix, self.left_camera.distortion_coefficients, \
        self.right_camera.camera_matrix, self.right_camera.distortion_coefficients,\
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objectPoints=self.left_camera.object_points,
            imagePoints1=self.left_camera.image_points,
            imagePoints2=self.right_camera.image_points,
            cameraMatrix1=self.left_camera.camera_matrix,
            distCoeffs1=self.left_camera.distortion_coefficients,
            cameraMatrix2=self.right_camera.camera_matrix,
            distCoeffs2=self.right_camera.distortion_coefficients,
            imageSize=self.image_size,
        )

    def get_rectify_map(self):
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            cameraMatrix1=self.left_camera.camera_matrix,
            distCoeffs1=self.left_camera.distortion_coefficients,
            cameraMatrix2=self.right_camera.camera_matrix,
            distCoeffs2=self.right_camera.distortion_coefficients,
            imageSize=self.image_size,
            R=self.R,
            T=self.T
        )

        self.left_maps = cv2.initUndistortRectifyMap(
            cameraMatrix=self.left_camera.camera_matrix,
            distCoeffs=self.left_camera.distortion_coefficients,
            R=self.R1,
            newCameraMatrix=self.P1,
            size=(640, 480),
            m1type=cv2.CV_16SC2
        )

        self.right_maps = cv2.initUndistortRectifyMap(
            cameraMatrix=self.right_camera.camera_matrix,
            distCoeffs=self.right_camera.distortion_coefficients,
            R=self.R2,
            newCameraMatrix=self.P2,
            size=(640, 480),
            m1type=cv2.CV_16SC2
        )


    def rectify(self, image, is_left_image):
        if is_left_image:
            maps = self.left_maps
        else:
            maps = self.right_maps

        rectified_image = cv2.remap(
            src=image,
            map1=maps[0],
            map2=maps[1],
            interpolation=cv2.INTER_LANCZOS4
        )

        return rectified_image




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


