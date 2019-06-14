import cv2
import numpy as np
import os

class Camera():
    def __init__(self,
                 camera_matrix = None,
                 distortion_coefficients = None,
                 rotation_matrix = None,
                 translation_vector = None,
                 pattern_size=(9,6),
                 image_size = (640,480),
                 image_root_dir = None
                 ):
        self.pattern_size = pattern_size                               # pattern size
        self.object_points = []                                        # object points
        self.image_points = []                                         # image points
        self.image_size = image_size                                   # image size
        self.camera_matrix = camera_matrix                             # the intrinsic matrix of the camera
        self.distortion_coefficients = distortion_coefficients         # the distortion coefficients [k1,k2,p1,p2,k3]
        self.rotation_matrix= rotation_matrix                          # rotation matrix R
        self.translation_vector = translation_vector                   # translation vectort t
        self.image_root_dir = image_root_dir                           # the root dir of images used for calibration
        self.image_paths = []                                          # the path of images used for calibration
        self.image_names = []                                          # the name of images used for calibration
        self.error = 0                                                 # the calibration error

    # calibrate the camera using the images in self.image_root_dir
    def calibrate(self):
        self.object_points = []
        self.image_points = []

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        object_points = np.zeros((6 * 9, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        self.image_paths = []
        for image_name in os.listdir(self.image_root_dir):
            if image_name[-4:] == '.jpg':
                image_path = os.path.join(self.image_root_dir, image_name)
                self.image_paths.append(image_path)
                self.image_names.append(image_name)

        self.image_paths.sort()
        self.image_names.sort()

        for image_path in self.image_paths:
            image = cv2.imread(image_path)
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




        self.error, self.camera_matrix, self.distortion_coefficients, self.rotation_matrix, self.translation_vector = cv2.calibrateCamera(
            objectPoints=self.object_points,
            imagePoints=self.image_points,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        return self.camera_matrix, self.distortion_coefficients, self.rotation_matrix, self.translation_vector

    # given a image, undistort the image and return the image before distortion
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

        return undistorted_image



