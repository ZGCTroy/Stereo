import os
import numpy as np
from Camera import Camera
from StereoCamera import StereoCamera
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create a stereo camera system
    stereo_camera =StereoCamera(
        left_camera=Camera(image_root_dir='../images/left'),
        right_camera=Camera(image_root_dir='../images/right'),
    )

    # stereo calibrate
    stereo_camera.calibrate()

    # calculate the map function of the undistortion and rectify
    stereo_camera.get_rectify_map()

    # Rectify
    for i in range(len(stereo_camera.left_camera.image_names)):
        left_image = cv2.imread(stereo_camera.left_camera.image_paths[i])
        right_image = cv2.imread(stereo_camera.right_camera.image_paths[i])

        rectified_left_image = stereo_camera.rectify(image=left_image, is_left_image=True)
        rectified_right_image = stereo_camera.rectify(image=right_image, is_left_image=False)

        cv2.imwrite(os.path.join('../images/rectified_left/', 'rectified_' +stereo_camera.left_camera.image_names[i]), rectified_left_image)
        cv2.imwrite(os.path.join('../images/rectified_right/', 'rectified_'+stereo_camera.right_camera.image_names[i]),rectified_right_image)

        plt.subplot(2, 2, 1)
        plt.imshow(left_image)
        plt.title('original left image')
        plt.subplot(2, 2, 2)
        plt.imshow(right_image)
        plt.title('original right image')
        plt.subplot(2, 2, 3)
        plt.imshow(rectified_left_image)
        plt.title('rectified left image')
        plt.subplot(2, 2, 4)
        plt.imshow(rectified_right_image)
        plt.title('rectified right image')
        #plt.show()

    print('P1 = ')
    print(stereo_camera.P1,'\n')

    print('P2 = ')
    print(stereo_camera.P2,'\n')

    print('Q')
    print(stereo_camera.Q,'\n')

    print('f = ')
    print(stereo_camera.P2[0][0],'\n')

    print('b * f = ')
    print(stereo_camera.P2[0][3],'\n')
    print('b = ')
    print(stereo_camera.P2[0][3]/stereo_camera.P2[0][0])

    print((stereo_camera.left_camera.camera_matrix[0][0]+stereo_camera.right_camera.camera_matrix[0][0])/2)
    print((stereo_camera.left_camera.camera_matrix[1][1] + stereo_camera.right_camera.camera_matrix[1][1]) / 2)
    print((stereo_camera.left_camera.camera_matrix[1][2] + stereo_camera.right_camera.camera_matrix[1][2]) / 2)