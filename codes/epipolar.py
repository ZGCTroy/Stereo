import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt




def drawlines(image, lines, points, colors):
    r, c = image.shape[:2]

    for line, point ,color in zip(lines, points, colors):
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
        image = cv2.line(image, (x0, y0), (x1, y1), color, 2)
        image = cv2.circle(image, tuple(point), 5, color, -1)

    return image

def draw_epipolar_lines(origin_stereo_camera, rectified_stereo_camera):

    for i in range(1, 14):
        imgL = cv2.imread(os.path.join('../images/left', 'left' + str(i).zfill(2) + '.jpg'))
        imgR = cv2.imread(os.path.join('../images/right', 'right' + str(i).zfill(2) + '.jpg'))
        rectified_imgL = cv2.imread(os.path.join('../images/rectified_left', 'rectified_left' + str(i).zfill(2) + '.jpg'))
        rectified_imgR = cv2.imread(os.path.join('../images/rectified_right', 'rectified_right' + str(i).zfill(2) + '.jpg'))

        colors = []
        for j in range(len(origin_stereo_camera.left_camera.image_points[i-1])):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))

        lines1 = cv2.computeCorrespondEpilines(origin_stereo_camera.right_camera.image_points[i-1].reshape(-1,1,2), 2,origin_stereo_camera.F)
        epipolar_imgL = drawlines(
            imgL,
            lines1.reshape(-1,3),
            origin_stereo_camera.left_camera.image_points[i-1].reshape(-1,2),
            colors
        )

        lines2 = cv2.computeCorrespondEpilines(origin_stereo_camera.left_camera.image_points[i-1].reshape(-1, 1, 2),1, origin_stereo_camera.F)

        epipolar_imgR = drawlines(
            imgR,
            lines2.reshape(-1, 3),
            origin_stereo_camera.right_camera.image_points[i-1].reshape(-1, 2),
            colors
        )

        lines1 = cv2.computeCorrespondEpilines(rectified_stereo_camera.right_camera.image_points[i - 1].reshape(-1, 1, 2),
                                               2, rectified_stereo_camera.F)
        epipolar_rectified_imgL = drawlines(
            rectified_imgL,
            lines1.reshape(-1, 3),
            rectified_stereo_camera.left_camera.image_points[i - 1].reshape(-1, 2),
            colors
        )

        lines2 = cv2.computeCorrespondEpilines(rectified_stereo_camera.left_camera.image_points[i - 1].reshape(-1, 1, 2),
                                               1, rectified_stereo_camera.F)

        epipolar_rectified_imgR = drawlines(
            rectified_imgR,
            lines2.reshape(-1, 3),
            rectified_stereo_camera.right_camera.image_points[i - 1].reshape(-1, 2),
            colors
        )


        # plt.subplot(2,2,1)
        # plt.imshow(epipolar_imgL)
        # plt.title(i)
        # plt.subplot(2, 2, 2)
        # plt.imshow(epipolar_imgR)
        # plt.subplot(2, 2, 3)
        # plt.imshow(epipolar_rectified_imgL)
        # plt.title(i)
        # plt.subplot(2, 2, 4)
        # plt.imshow(epipolar_rectified_imgR)
        # plt.show()

        cv2.imwrite(
            os.path.join('../images/epipolar_left/', 'epipolar_left' + str(i).zfill(2) + '.jpg'),
            epipolar_imgL)

        cv2.imwrite(
            os.path.join('../images/epipolar_right/', 'epipolar_right' + str(i).zfill(2) + '.jpg'),
            epipolar_imgR)

        cv2.imwrite(
            os.path.join('../images/epipolar_rectified_left/', 'epipolar_rectified_left' + str(i).zfill(2) + '.jpg' ),
                    epipolar_rectified_imgL)

        cv2.imwrite(
            os.path.join('../images/epipolar_rectified_right/', 'epipolar_rectified_right' + str(i).zfill(2) + '.jpg'),
            epipolar_rectified_imgR)

if __name__ == '__main__':
    # create a stereo camera system
    origin_stereo_camera = stereo_camera_system(
        left_camera=single_camera_system(image_root_dir='../images/left'),
        right_camera=single_camera_system(image_root_dir='../images/right'),
    )

    rectified_stereo_camera = stereo_camera_system(
        left_camera=single_camera_system(image_root_dir='../images/rectified_left'),
        right_camera=single_camera_system(image_root_dir='../images/rectified_right'),
    )

    # stereo calibrate
    origin_stereo_camera.stereo_calibrate()
    rectified_stereo_camera.stereo_calibrate()

    # calculate the map function of the undistortion and rectify
    origin_stereo_camera.calculate_undistort_and_rectify_map()
    rectified_stereo_camera.calculate_undistort_and_rectify_map()

    draw_epipolar_lines(
        origin_stereo_camera=origin_stereo_camera,
        rectified_stereo_camera=rectified_stereo_camera
    )