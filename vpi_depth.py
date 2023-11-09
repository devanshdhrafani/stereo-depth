#!/usr/bin/env python
import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import vpi
import numpy as np
import yaml

def load_calibration_data(calibration_file):
    with open(calibration_file, 'r') as f:
        calibration_data = yaml.load(f, Loader=yaml.FullLoader)
    return calibration_data

def get_camera_properties(calibration_data):

    # import ipdb; ipdb.set_trace()
    # Intrinsic parameters
    K = [calibration_data['intrinsic']['projection_parameters']['fx'], 0, calibration_data['intrinsic']['projection_parameters']['cx'],
        0, calibration_data['intrinsic']['projection_parameters']['fy'], calibration_data['intrinsic']['projection_parameters']['cy'],
        0, 0, 1]

    # Distortion parameters (for Pinhole model)
    D = [calibration_data['intrinsic']['distortion_parameters']['k1'],
                     calibration_data['intrinsic']['distortion_parameters']['k2'],
                     calibration_data['intrinsic']['distortion_parameters']['p1'],
                     calibration_data['intrinsic']['distortion_parameters']['p2'],
                     0]  # k3 is 0 for a Pinhole camera

    # Extrinsic parameters (rotation and translation)
    rotation_matrix = np.array(calibration_data['extrinsic']['rotation_to_body']['data']).reshape(3, 3)
    translation_vector = np.array(calibration_data['extrinsic']['translation_to_body']['data']).reshape(3, 1)

    # The rotation matrix and translation vector are already in the correct format for ROS CameraInfo
    R = list(rotation_matrix.flatten())
    P = list((rotation_matrix @ translation_vector).flatten()) + [0, 0, 0, 1]

    # Convert K, D, rotation_matrix, translation_vector to cv2 matrices
    K = np.array(K).reshape(3, 3)
    D = np.array(D).reshape(1, 5)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
    translation_vector = np.array(translation_vector).reshape(3, 1)

    return K, D, rotation_matrix, translation_vector

class StereoDisparityCalculator:
    def __init__(self):
        self.bridge = CvBridge()
        self.disparity_publisher_1 = rospy.Publisher('/stereo/disparity_vpi', Image, queue_size=1)
        self.disparity_publisher_2 = rospy.Publisher('/stereo/disparity_opencv', Image, queue_size=1)

        left_image_sub = message_filters.Subscriber('/thermal_left/image', Image)
        right_image_sub = message_filters.Subscriber('/thermal_right/image', Image)

        self.ts = message_filters.TimeSynchronizer([left_image_sub, right_image_sub], 10)
        self.ts.registerCallback(self.process_images)

        # import yaml calibration files
        self.thermal_left_yaml = load_calibration_data("./thermal_left.yaml")
        self.thermal_right_yaml = load_calibration_data("./thermal_right.yaml")

        K_left, D_left, R_left, t_left = get_camera_properties(self.thermal_left_yaml)
        K_right, D_right, R_right, t_right = get_camera_properties(self.thermal_right_yaml)

        R = np.dot(R_right, R_left.T)  # Rotation matrix
        T = t_right - np.dot(R, t_left)  # Translation vector

        self.image_size = (640, 512)

        self.i = 0

        # Compute the rectification maps
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, self.image_size, R, T)

        # Compute the rectification maps for both cameras
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, self.image_size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, self.image_size, cv2.CV_32FC1)

    def process_images(self, left_image_msg, right_image_msg):
        # print encoding

        print("left_image_msg.encoding: ", left_image_msg.encoding)
        left_image_msg.encoding = "mono16"
        right_image_msg.encoding = "mono16"

        try:
            left_image = self.bridge.imgmsg_to_cv2(left_image_msg, desired_encoding='mono8')
            right_image = self.bridge.imgmsg_to_cv2(right_image_msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr("Error converting images: %s", e)
            return
        
        # rectified_stitched = np.hstack((left_image, right_image))
        # cv2.imshow("rectified_stitched", rectified_stitched)
        # if not cv2.waitKey(1):
        #     pass

        # Rectify the left and right images
        rectified_left_cv = cv2.remap(left_image, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectified_right_cv = cv2.remap(right_image, self.map2x, self.map2y, cv2.INTER_LINEAR)

        # Compute stereo disparity using Nvidia VPI package
        disparity_image = vpi.Image(self.image_size, vpi.Format.U8)

        # stich rectified images side by side
        unrectified_stitched = np.hstack((left_image, right_image))
        rectified_stitched = np.hstack((rectified_left_cv, rectified_right_cv))

        all_stitched = np.vstack((unrectified_stitched, rectified_stitched))

        # cv2.imshow("rectified_stitched", rectified_stitched)
        # cv2.waitKey(1)

        # cv2.imwrite(f"rectified_images/all_stitched_{self.i}.png", all_stitched)

        # convert to vpi image
        rectified_left = vpi.asimage(rectified_left_cv)
        rectified_right = vpi.asimage(rectified_right_cv)

        unrectified_left = vpi.asimage(left_image)
        unrectified_right = vpi.asimage(right_image)

        with vpi.Backend.CUDA:
            disparity_image = vpi.stereodisp(unrectified_left, unrectified_right, window=100, 
                                    maxdisp=64).convert(vpi.Format.U8, scale=1.0/(32*64)*255)

        with disparity_image.rlock_cpu() as outData:
            disparity_data = outData
            disparity_msg = self.bridge.cv2_to_imgmsg(disparity_data, encoding='mono8')

        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
        disparity_opencv = stereo.compute(rectified_left_cv,rectified_right_cv)

        # Publish the computed disparity image
        self.disparity_publisher_1.publish(disparity_msg)
        self.disparity_publisher_2.publish(self.bridge.cv2_to_imgmsg(disparity_opencv, encoding='16SC1'))

        self.i += 1

def main():
    rospy.init_node('stereo_disparity_node')
    stereo_calculator = StereoDisparityCalculator()
    rospy.spin()

if __name__ == '__main__':
    main()
