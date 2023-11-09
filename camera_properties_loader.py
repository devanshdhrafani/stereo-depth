import numpy as np
import yaml

class CameraPropertiesLoader:
    def __init__(self, calibration_file):
        self.K = None
        self.D = None
        self.R = None
        self.t = None

        calibration_data = self.load_calibration_data(calibration_file)
        self.process_calibration(calibration_data)

    def get_properties(self):
        return self.K, self.D, self.R, self.t

    def load_calibration_data(self, calibration_file):
        with open(calibration_file, 'r') as f:
            calibration_data = yaml.load(f, Loader=yaml.FullLoader)
        return calibration_data

    def process_calibration(self, calibration_data):
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

        self.K = K
        self.D = D
        self.R = rotation_matrix
        self.t = translation_vector