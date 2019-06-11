import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def single_camera_calibration(images_root_dir):
    # set image paths
    image_paths = []
    for file_name in os.listdir('./images/left'):
        image_paths.append('./images/left/' + file_name)

    # calibration
    total_object_points = []
    total_image_points = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    object_points = np.zeros((6 * 9, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    success = 0
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        found, image_points = cv2.findChessboardCorners(
            image=gray_image,
            patternSize=(9, 6),
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

            total_object_points.append(object_points)
            total_image_points.append(image_points)

    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        objectPoints=total_object_points,
        imagePoints=total_image_points,
        imageSize=(640, 480),
        cameraMatrix=None,
        distCoeffs=None
    )

    return camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors

def undistort(image, camera_matrix, distortion_coefficients):
    print(type(image))
    plt.imshow(image)
    plt.show()

    h = image.shape[0]
    w = image.shape[1]
    print(image.shape)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (w, h),
        1,
        (w, h)
    )

    undistorted_image = cv2.undistort(
        src=image,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion_coefficients,
        dst=None,
        newCameraMatrix=new_camera_matrix
    )

    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y + h, x:x + w]

    return undistorted_image

if __name__ == '__main__':
    camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = single_camera_calibration('./images/left')

    image=cv2.imread('./images/left/left02.jpg')
    undistorted_image = undistort(image,camera_matrix,distortion_coefficients)

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(undistorted_image)
    plt.show()