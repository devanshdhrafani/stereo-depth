from camera_properties_loader import CameraPropertiesLoader
import numpy as np
import cv2


class RectifyImage(object):
    def __init__(self, left_calibration_file, right_calibration_file):
        self.left_properties = CameraPropertiesLoader(left_calibration_file)
        self.right_properties = CameraPropertiesLoader(right_calibration_file)

    def __call__(self, left_raw, right_raw):
        K_left, D_left, R_left, t_left = self.left_properties.get_properties()
        K_right, D_right, R_right, t_right = self.right_properties.get_properties()

        R = np.dot(R_right, R_left.T)  # Rotation matrix
        T = t_right - np.dot(R, t_left)  # Translation vector

        baseline = np.abs(T[1])
        focal_length = (K_left[0][0] + K_left[1][1]) / 2

        image_size = left_raw.shape[1], left_raw.shape[0]

        # Compute the rectification maps
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, D_left, K_right, D_right, image_size, R, T
        )

        map1x, map1y = cv2.initUndistortRectifyMap(
            K_left, D_left, R1, P1, image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K_right, D_right, R2, P2, image_size, cv2.CV_32FC1
        )

        left_rectified = cv2.remap(left_raw, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_raw, map2x, map2y, cv2.INTER_LINEAR)

        return left_rectified, right_rectified, baseline, focal_length
