import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

from Camera import Camera

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
        self.left_camera = left_camera                      # Camera Class, the left camera
        self.right_camera = right_camera                    # Camera Class, the right camera
        self.image_size = image_size                        # image size
        self.R = R                                          # rotation matrix from the left to the right camera coordinate
        self.T = T                                          # translation vectort from the left to the right camera coordinate
        self.E = E                                          # essential matrix
        self.F = F                                          # fundamental matrix
        self.left_maps = None                               # the rectify mapping for the left camera
        self.right_maps = None                              # the rectify mapping for the right camera
        self.P1 = None                                      # the new projection matrix for the left camera
        self.P2 = None                                      # the new projection matrix for the right camera
        self.R1 = None                                      # the left transformation matrix from the old to the new
        self.R2 = None                                      # the right transformation matrix from the old to the new
        self.Q = None                                       # the disparity-to-depth transformation matrix
        self.calibrate_error = 0                            # the error of stereo calibration

    # calibrate the stereo camera system
    def calibrate(self):

        self.left_camera.calibrate()
        self.right_camera.calibrate()
        
        self.calibrate_error, self.left_camera.camera_matrix, self.left_camera.distortion_coefficients, \
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

    # calculate the undistort and rectify map
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

    # use the map calculated by get_rectify_map() to rectify a image
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


