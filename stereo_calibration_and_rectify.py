import cv2
from matplotlib import pyplot as plt
import numpy as np
import os



def get_objectPoints_and_imagePoints(left_images_path, right_images_path, is_shown=False):
    # calibration
    total_object_points = []
    total_left_image_points = []
    total_right_image_points = []

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    pattern_size = (9, 6)
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    images_id = []

    # images
    for i in range(1,14):
        left_image = cv2.imread(left_images_path + '/left' + str(i).zfill(2) + '.jpg')
        right_image = cv2.imread(right_images_path + '/right' + str(i).zfill(2) + '.jpg')

        left_gray_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        left_found, left_image_points = cv2.findChessboardCorners(
            image=left_gray_image,
            patternSize=pattern_size,
            corners=None,
        )
        right_found, right_image_points = cv2.findChessboardCorners(
            image=right_gray_image,
            patternSize=pattern_size,
            corners=None,
        )

        # If found, add object points, image points (after refining them)
        if left_found and right_found:
            left_image_points = cv2.cornerSubPix(
                image=left_gray_image,
                corners=left_image_points,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.01
                )
            )
            right_image_points = cv2.cornerSubPix(
                image=right_gray_image,
                corners=right_image_points,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.01
                )
            )

            images_id.append(i)
            total_object_points.append(object_points)
            total_left_image_points.append(left_image_points)
            total_right_image_points.append(right_image_points)

            if is_shown:
                cv2.drawChessboardCorners(
                    image=left_image,
                    patternSize=pattern_size,
                    corners=left_image_points,
                    patternWasFound=left_found
                )
                cv2.drawChessboardCorners(
                    image=right_image,
                    patternSize=pattern_size,
                    corners=right_image_points,
                    patternWasFound=right_found
                )
                plt.subplot(1,2,1)
                plt.imshow(left_image)
                plt.subplot(1,2,2)
                plt.imshow(right_image)
                plt.show()


    return images_id, total_object_points, total_left_image_points, total_right_image_points


def stereo_calibration(object_points, left_image_points, right_image_points):
    # single camera calibration for the left camera
    _, left_camera_matrix, left_distortion_coefficients, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=left_image_points,
        imageSize=(640, 480),
        cameraMatrix=None,
        distCoeffs=None
    )

    # single camera calibration for the left camera
    _, right_camera_matrix, right_distortion_coefficients, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=right_image_points,
        imageSize=(640, 480),
        cameraMatrix=None,
        distCoeffs=None
    )

    # stereo camera calibration
    retval, left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=object_points,
        imagePoints1=left_image_points,
        imagePoints2=right_image_points,
        cameraMatrix1=left_camera_matrix,
        distCoeffs1=left_distortion_coefficients,
        cameraMatrix2=right_camera_matrix,
        distCoeffs2=right_distortion_coefficients,
        imageSize=(640, 480),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    return left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, R, T, E, F



def get_undistort_and_rectify_map(
        left_camera_matrix,
        left_distortion_coefficients,
        right_camera_matrix,
        right_distortion_coefficients,
        R1,
        P1,
        R2,
        P2
):
    left_maps = cv2.initUndistortRectifyMap(
        cameraMatrix=left_camera_matrix,
        distCoeffs=left_distortion_coefficients,
        R=R1,
        newCameraMatrix=P1,
        size=(640, 480),
        m1type=cv2.CV_16SC2
    )

    right_maps = cv2.initUndistortRectifyMap(
        cameraMatrix=right_camera_matrix,
        distCoeffs=right_distortion_coefficients,
        R=R2,
        newCameraMatrix=P2,
        size=(640, 480),
        m1type=cv2.CV_16SC2
    )

    return left_maps, right_maps


def generate_undistorted_and_rectified_images(left_images_path = './images/left',
                                              right_images_path = './images/right',
                                              rectify_left_images_path = './images/rectify_left',
                                              rectify_right_images_path = './images/rectify_right'
                                              ):
    # get object points and image points
    images_id, object_points, left_image_points, right_image_points = get_objectPoints_and_imagePoints(
        left_images_path='./images/left',
        right_images_path='./images/right',
        is_shown=False
    )

    # calculate two camera matrixes , distortion coefficients and R,T,E,F
    left_camera_matrix, left_distortion_coefficients, \
    right_camera_matrix, right_distortion_coefficients, R, T, E, F = stereo_calibration(
        object_points=object_points,
        left_image_points=left_image_points,
        right_image_points=right_image_points
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=left_camera_matrix,
        distCoeffs1=left_distortion_coefficients,
        cameraMatrix2=right_camera_matrix,
        distCoeffs2=right_distortion_coefficients,
        imageSize=(640, 480),
        R=R,
        T=T
    )

    left_maps, right_maps = get_undistort_and_rectify_map(
        left_camera_matrix=left_camera_matrix,
        left_distortion_coefficients=left_distortion_coefficients,
        right_camera_matrix=right_camera_matrix,
        right_distortion_coefficients=right_distortion_coefficients,
        R1=R1,
        P1=P1,
        R2=R2,
        P2=P2
    )

    for i in range(1,14):
        left_image = cv2.imread(left_images_path + '/left' + str(i).zfill(2) + '.jpg')
        right_image = cv2.imread(right_images_path + '/right' + str(i).zfill(2) + '.jpg')

        left_image_remap = cv2.remap(
            src=left_image,
            map1=left_maps[0],
            map2=left_maps[1],
            interpolation=cv2.INTER_LANCZOS4
        )
        right_image_remap = cv2.remap(
            src=right_image,
            map1=right_maps[0],
            map2=right_maps[1],
            interpolation=cv2.INTER_LANCZOS4
        )

        cv2.imwrite(rectify_left_images_path + '/left' + str(i+1).zfill(2) + '.jpg', left_image_remap)
        cv2.imwrite(rectify_right_images_path + '/right' + str(i+1).zfill(2) + '.jpg', right_image_remap)


def generate_undistorted_images(left_images_path = './images/left',
                                              right_images_path = './images/right',
                                              undistorted_left_images_path = './images/undistorted_left',
                                              undistorted_right_images_path = './images/undistorted_right'
                                              ):
    images_id, object_points, left_image_points, right_image_points = get_objectPoints_and_imagePoints(
        left_images_path=left_images_path,
        right_images_path=right_images_path,
        is_shown=False
    )

    _, left_camera_matrix, left_distortion_coefficients, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=left_image_points,
        imageSize=(640, 480),
        cameraMatrix=None,
        distCoeffs=None
    )

    # single camera calibration for the left camera
    _, right_camera_matrix, right_distortion_coefficients, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=right_image_points,
        imageSize=(640, 480),
        cameraMatrix=None,
        distCoeffs=None
    )

    for i in range(1,14):
        left_image = cv2.imread(left_images_path + '/left' + str(i).zfill(2) + '.jpg')
        right_image = cv2.imread(right_images_path + '/right' + str(i).zfill(2) + '.jpg')

        h, w = left_image.shape[:2]
        new_left_camera_matrix, left_roi = cv2.getOptimalNewCameraMatrix(
            left_camera_matrix,
            left_distortion_coefficients,
            (w, h),
            1,
            (w, h)
        )

        undistorted_left_image = cv2.undistort(
            src=left_image,
            cameraMatrix=left_camera_matrix,
            distCoeffs=left_distortion_coefficients,
            dst=None,
            newCameraMatrix=new_left_camera_matrix
        )

        x, y, w, h = left_roi
        undistorted_left_image = undistorted_left_image[y:y + h, x:x + w]

        h, w = right_image.shape[:2]
        new_right_camera_matrix, right_roi = cv2.getOptimalNewCameraMatrix(
            right_camera_matrix,
            right_distortion_coefficients,
            (w, h),
            1,
            (w, h)
        )

        undistorted_right_image = cv2.undistort(
            src=right_image,
            cameraMatrix=right_camera_matrix,
            distCoeffs=right_distortion_coefficients,
            dst=None,
            newCameraMatrix=new_right_camera_matrix
        )

        x, y, w, h = right_roi
        undistorted_right_image = undistorted_right_image[y:y + h, x:x + w]

        cv2.imwrite(undistorted_left_images_path + '/left' + str(i).zfill(2) + '.jpg', undistorted_left_image)
        cv2.imwrite(undistorted_right_images_path + '/right' + str(i).zfill(2) + '.jpg', undistorted_right_image)

def drawlines(image, lines, points, colors):
    r, c = image.shape[:2]

    for line, point ,color in zip(lines, points, colors):
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        image = cv2.line(image, (x0, y0), (x1, y1), color, 2)
        image = cv2.circle(image, tuple(point), 5, color, -1)

    return image

def draw_epipolar_lines(left_images_paths,right_images_paths):

    images_id, object_points, left_image_points, right_image_points = get_objectPoints_and_imagePoints(
        left_images_path=left_images_paths,
        right_images_path=right_images_paths,
        is_shown=False
    )

    left_camera_matrix, left_distortion_coefficients, \
    right_camera_matrix, right_distortion_coefficients, R, T, E, F = stereo_calibration(
        object_points=object_points,
        left_image_points=left_image_points,
        right_image_points=right_image_points
    )

    for i in range(len(images_id)):
        left_image = cv2.imread(left_images_paths + '/left' + str(images_id[i]).zfill(2) + '.jpg')
        right_image = cv2.imread(right_images_paths + '/right' + str(images_id[i]).zfill(2) + '.jpg')

        colors = []
        for j in range(len(right_image_points[i])):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))

        lines1 = cv2.computeCorrespondEpilines(right_image_points[i].reshape(-1,1,2), 2,F)
        epipolar_left_image = drawlines(
            left_image,
            lines1.reshape(-1,3),
            left_image_points[i].reshape(-1,2),
            colors
        )

        lines2 = cv2.computeCorrespondEpilines(left_image_points[i].reshape(-1, 1, 2),1, F)
        epipolar_right_image = drawlines(
            right_image,
            lines2.reshape(-1, 3),
            right_image_points[i].reshape(-1,2),
            colors
        )


        plt.subplot(1,2,1)
        plt.imshow(epipolar_left_image)
        plt.title(i)
        plt.subplot(1, 2, 2)
        plt.imshow(epipolar_right_image)
        plt.show()


if __name__ == '__main__':

    pattern_size = (9,6)

    images_id, object_points, left_image_points, right_image_points = get_objectPoints_and_imagePoints(
        left_images_path='./images/left',
        right_images_path='./images/right',
        is_shown=False
    )

    # calculate two camera matrixes , distortion coefficients and R,T,E,F
    left_camera_matrix, left_distortion_coefficients, \
    right_camera_matrix, right_distortion_coefficients, R, T, E, F = stereo_calibration(
        object_points=object_points,
        left_image_points=left_image_points,
        right_image_points=right_image_points
    )

    print(T)

    # generate_undistorted_images()
    # draw_epipolar_lines(
    #     left_images_paths='./images/undistorted_left',
    #     right_images_paths='./images/undistorted_right'
    # )
    #
    # generate_undistorted_and_rectified_images()
    # draw_epipolar_lines(
    #     left_images_paths='./images/rectify_left',
    #     right_images_paths='./images/rectify_right'
    # )

    # get object points and image points
    # images_id, object_points, left_image_points, right_image_points = get_objectPoints_and_imagePoints(
    #     left_images_path='./images/rectify_left',
    #     right_images_path='./images/rectify_right',
    #     is_shown=False
    # )
    #
    # # calculate two camera matrixes , distortion coefficients and R,T,E,F
    # left_camera_matrix, left_distortion_coefficients, \
    # right_camera_matrix, right_distortion_coefficients, R, T, E, F = stereo_calibration(
    #     object_points=object_points,
    #     left_image_points=left_image_points,
    #     right_image_points=right_image_points
    # )
    #
    # print(T)
    # left_camera_matrix = left_camera_matrix
    #
    # imgL = cv2.imread('./images/rectify_left/left03.jpg', 0)
    # imgR = cv2.imread('./images/rectify_right/right03.jpg', 0)
    #
    # stereo = cv2.StereoBM_create(numDisparities=48, blockSize=15)
    # disparity = stereo.compute(imgL, imgR)
    # #disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    # window_size = 3
    # min_disp = 16
    # num_disp = 112 - min_disp
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    #                               numDisparities=num_disp,
    #                               blockSize=16,
    #                               P1=8 * 3 * window_size ** 2,
    #                               P2=32 * 3 * window_size ** 2,
    #                               disp12MaxDiff=1,
    #                               uniquenessRatio=10,
    #                               speckleWindowSize=100,
    #                               speckleRange=32
    #                               )
    # disparity2 = stereo.compute(imgL, imgR)
    #
    # plt.subplot(2, 2, 1)
    # plt.imshow(imgL)
    # plt.subplot(2, 2, 2)
    # plt.imshow(imgR)
    # plt.subplot(2, 2, 3)
    # plt.imshow(disparity)
    # plt.subplot(2, 2, 4)
    # plt.imshow(disparity2)
    # plt.show()
    #
    # window_size = 3
    # min_disp = 16
    # num_disp = 112 - min_disp
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    #                                numDisparities=num_disp,
    #                                blockSize=16,
    #                                P1=8 * 3 * window_size ** 2,
    #                                P2=32 * 3 * window_size ** 2,
    #                                disp12MaxDiff=1,
    #                                uniquenessRatio=10,
    #                                speckleWindowSize=100,
    #                                speckleRange=32
    #                                )
    # disparity2 = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # disparity2 = cv2.normalize(disparity2, disparity2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # plt.subplot(2, 2, 1)
    # plt.imshow(imgL)
    # plt.subplot(2, 2, 2)
    # plt.imshow(imgR)
    # plt.subplot(2, 2, 3)
    # plt.imshow(disparity)
    # plt.subplot(2, 2, 4)
    # plt.imshow(disparity2)
    # plt.show()
    #
    # cv2.imshow("123",disparity2)
    # cv2.waitKey()


