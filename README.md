The repository is the implement code of the Stereo Project. 

[Here](https://cn.overleaf.com/read/qhzdxpsksbcs) is the report of the project, which is written in latex in online overleaf.

# Structure
* code
    * Class file
        * Camera.py
        * StereoCamera.py
        
    * Test file for testing the function of class
        * test_calibrate.py
        * test_undistort.py
        * test_stereoCalibrate.py
        * test_rectify.py
        * test_epipolar.py
        * test_SGBM.py
    
* env

    the python Virtualenv, which can be loaded by most IDEs such as Pycharm, Eclipse

* images
    * disparity

        The disparity map of each image

    * epipolar_rectified_left

        The rectified left images with epipolar lines

    * epipolar_rectified_right

        The rectified right images with epipolar lines