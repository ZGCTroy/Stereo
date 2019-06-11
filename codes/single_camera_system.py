import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class single_camera_system():
    def __init__(self,
                 camera_matrix = None,
                 distortion_coefficients = None,
                 rotation_matrix = None,
                 translation_vector = None,
                 pattern_size=(9,6),
                 image_size = (640,480),
                 image_root_dir = None
                 ):
        self.pattern_size = pattern_size
        self.object_points = []
        self.image_points = []
        self.image_size = image_size
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.rotation_matrix= rotation_matrix
        self.translation_vector = translation_vector
        self.image_root_dir = image_root_dir
        self.image_paths = []
        self.image_names = []

    def calibrate(self):
        self.object_points = []
        self.image_points = []

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        object_points = np.zeros((6 * 9, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        self.image_paths = []
        for image_name in os.listdir(self.image_root_dir):
            image_path = os.path.join(self.image_root_dir, image_name)
            self.image_paths.append(image_path)
            self.image_names.append(image_name)

        self.image_paths.sort()
        self.image_names.sort()

        for image_path in self.image_paths:

            image = cv2.imread(image_path)
            print(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            found, image_points = cv2.findChessboardCorners(
                image=gray_image,
                patternSize=self.pattern_size,
                corners=None,
            )

            # If found, add object points, image points (after refining them)
            if found:
                image_points = cv2.cornerSubPix(
                    image=gray_image,
                    corners=image_points,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.01
                    )
                )
                self.object_points.append(object_points)
                self.image_points.append(image_points)
            else:
                print(image_path, 'fails')




        ret, self.camera_matrix, self.distortion_coefficients, self.rotation_matrix, self.translation_vector = cv2.calibrateCamera(
            objectPoints=self.object_points,
            imagePoints=self.image_points,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        return self.camera_matrix, self.distortion_coefficients, self.rotation_matrix, self.translation_vector

    def undistort(self, image):
        h = image.shape[0]
        w = image.shape[1]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.distortion_coefficients,
            imageSize=(w, h),
            alpha=1,
            newImgSize=(w, h)
        )

        undistorted_image = cv2.undistort(
            src=image,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.distortion_coefficients,
            dst=None,
            newCameraMatrix=new_camera_matrix
        )

        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y + h, x:x + w]

        return undistorted_image



