import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

from camera import Camera


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


class stereo_camera_system():
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


    def stereo_calibrate(self):

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

    def calculate_undistort_and_rectify_map(self):
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


    def undistort_and_rectify_image(self, image, is_left_image):
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
    stereo_camera = stereo_camera_system(
        left_camera=single_camera_system(image_root_dir='../images/left'),
        right_camera=single_camera_system(image_root_dir='../images/right'),
    )

    # stereo calibrate
    stereo_camera.stereo_calibrate()

    # calculate the map function of the undistortion and rectify
    stereo_camera.calculate_undistort_and_rectify_map()
    #
    # for i in range(1, 14):
    #     left_image = cv2.imread(left_images_path + '/left' + str(i).zfill(2) + '.jpg')
    #     right_image = cv2.imread(right_images_path + '/right' + str(i).zfill(2) + '.jpg')
    #
    #     left_image_remap = cv2.remap(
    #         src=left_image,
    #         map1=left_maps[0],
    #         map2=left_maps[1],
    #         interpolation=cv2.INTER_LANCZOS4
    #     )
    #     right_image_remap = cv2.remap(
    #         src=right_image,
    #         map1=right_maps[0],
    #         map2=right_maps[1],
    #         interpolation=cv2.INTER_LANCZOS4
    #     )
    #
    #     cv2.imwrite(rectify_left_images_path + '/left' + str(i + 1).zfill(2) + '.jpg', left_image_remap)
    #     cv2.imwrite(rectify_right_images_path + '/right' + str(i + 1).zfill(2) + '.jpg', right_image_remap)

    #

