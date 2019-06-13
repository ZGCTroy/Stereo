import os
import numpy as np
from single_camera_system import single_camera_system
from stereo_camera_system import stereo_camera_system
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # create a stereo camera system
    stereo_camera = stereo_camera_system(
        left_camera=single_camera_system(image_root_dir='../images/left'),
        right_camera=single_camera_system(image_root_dir='../images/right'),
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

    print(stereo_camera.P1)

    print(stereo_camera.P2)

    print(stereo_camera.T)