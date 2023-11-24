#!/usr/bin/env python
import cv2
import vpi
import numpy as np
from camera_properties_loader import CameraPropertiesLoader
from tqdm import tqdm


def nothing(x):
    pass


class StereoDepthInference:
    def __init__(self, left_camera_properties, right_camera_properties, image_size):
        K_left, D_left, R_left, t_left = left_camera_properties.get_properties()
        K_right, D_right, R_right, t_right = right_camera_properties.get_properties()

        R = np.dot(R_right, R_left.T)  # Rotation matrix
        T = t_right - np.dot(R, t_left)  # Translation vector

        self.baseline = np.abs(T[1])
        self.focal_length = self.calculate_focal_length(K_left)

        self.image_size = image_size

        # Compute the rectification maps
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_left, D_left, K_right, D_right, self.image_size, R, T
        )

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K_left, D_left, R1, P1, self.image_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            K_right, D_right, R2, P2, self.image_size, cv2.CV_32FC1
        )

    def calculate_focal_length(self, K):
        return (K[0][0] + K[1][1]) / 2

    def map_to_color(self, image):
        image = image.astype(np.float32)
        image = cv2.normalize(
            image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        # create a mask to capture only non-one values, then apply colormap
        mask = image == 0
        # print image pixel value at a point I clicked
        # def click_event(event, x, y, flags, params):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         print(image[y, x])

        # cv2.namedWindow("image")
        # cv2.imshow("image", image)
        # cv2.setMouseCallback("image", click_event)
        # cv2.waitKey(0)

        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        # for pixels with zero disparity, set color to black
        image[np.where(mask == True)] = [0, 0, 0]

        # image = vertical_interpolation(image)
        return image

    def disparity_to_depth(self, disparity_image):
        disparity_image = disparity_image.astype(np.float32)
        disparity_color = self.map_to_color(disparity_image)
        # cv2.imshow("disparity", disparity_color)
        # cv2.waitKey(0)

        depth_image = np.zeros_like(disparity_image)
        depth_image[disparity_image > 0] = (
            self.focal_length * self.baseline / disparity_image[disparity_image > 0]
        )
        # save depth image to txt file
        # np.savetxt("depth.txt", depth_image)

        # cv2.imshow("depth", depth_image)
        # cv2.waitKey(0)

        depth_image = self.map_to_color(depth_image)

        return depth_image

    def get_disparity(self, left, right, method="vpi"):
        # Convert to grayscale
        left_image = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).copy()
        right_image = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).copy()

        # Rectify the left and right images
        rectified_left_cv = cv2.remap(
            left_image, self.map1x, self.map1y, cv2.INTER_LINEAR
        )
        rectified_right_cv = cv2.remap(
            right_image, self.map2x, self.map2y, cv2.INTER_LINEAR
        )

        if method == "vpi":
            disparity_image, confidence_img = self.vpi_disparity(
                rectified_left_cv, rectified_right_cv
            )
        elif method == "opencv":
            disparity_image = self.opencv_disparity(
                rectified_left_cv, rectified_right_cv
            )
        elif method == "opencv_tuner":
            disparity_image = self.opencv_disparity_tuner(
                rectified_left_cv, rectified_right_cv
            )
        else:
            raise Exception("Invalid method")

        # stich rectified images
        frame_rectified = np.hstack((rectified_left_cv, rectified_right_cv))
        frame_rectified = cv2.cvtColor(frame_rectified, cv2.COLOR_GRAY2BGR)

        return (disparity_image, confidence_img, frame_rectified)

    def vpi_disparity(self, rectified_left, rectified_right):
        # cv2.imshow("rectified_left", rectified_left)
        # cv2.waitKey(0)

        with vpi.Backend.CUDA:
            left = vpi.asimage(rectified_left).convert(vpi.Format.Y16_ER, scale=1)
            right = vpi.asimage(rectified_right).convert(vpi.Format.Y16_ER, scale=1)

        disparity_image, disparity_image_cv = None, None

        confidenceU16 = vpi.Image((640, 512), vpi.Format.U16)

        maxdisp = 64

        with vpi.Backend.CUDA:
            # disparity_image = vpi.stereodisp(left, right, window=5, maxdisp=64).convert(vpi.Format.U8, scale=1.0/(32*64)*255)
            disparityS16 = vpi.stereodisp(
                left,
                right,
                downscale=1,
                out_confmap=confidenceU16,
                window=5,
                maxdisp=maxdisp,
                quality=8,
                # conftype=vpi.ConfidenceType.ABSOLUTE,
                confthreshold=500,
            )
            disparity_image = disparityS16.convert(
                vpi.Format.U8, scale=255.0 / (32 * maxdisp)
            ).cpu()
            confidenceU8 = confidenceU16.convert(
                vpi.Format.U8, scale=255.0 / 65535
            ).cpu()

        disparity_image = self.map_to_color(disparity_image)
        # depth_image = self.disparity_to_depth(disparity_image)
        # cv2.imshow("depth", depth_image)
        # cv2.waitKey(0)

        # with disparity_image.rlock_cpu() as outData:
        #     disparity_image_cv = outData

        # Scale disparity image to 0-255
        # disparity_image_cv = cv2.normalize(disparity_image_cv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # cv2.imshow("disparity", disparity_image)
        # cv2.waitKey(0)

        return disparity_image, confidenceU8

    def opencv_disparity(self, rectified_left, rectified_right):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity_image = stereo.compute(rectified_left, rectified_right).astype(
            np.float32
        )
        disparity_image = (disparity_image / 16.0 - 5) / 16

        disparity_image = cv2.normalize(
            disparity_image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        return disparity_image

    def opencv_disparity_tuner(self, rectified_left, rectified_right):
        # show left image
        cv2.imshow("left", rectified_left)

        cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("disp", 600, 600)

        cv2.createTrackbar("numDisparities", "disp", 1, 17, nothing)
        cv2.createTrackbar("blockSize", "disp", 5, 50, nothing)
        cv2.createTrackbar("preFilterType", "disp", 1, 1, nothing)
        cv2.createTrackbar("preFilterSize", "disp", 2, 25, nothing)
        cv2.createTrackbar("preFilterCap", "disp", 5, 62, nothing)
        cv2.createTrackbar("textureThreshold", "disp", 10, 100, nothing)
        cv2.createTrackbar("uniquenessRatio", "disp", 15, 100, nothing)
        cv2.createTrackbar("speckleRange", "disp", 0, 100, nothing)
        cv2.createTrackbar("speckleWindowSize", "disp", 3, 25, nothing)
        cv2.createTrackbar("disp12MaxDiff", "disp", 5, 25, nothing)
        cv2.createTrackbar("minDisparity", "disp", 5, 25, nothing)

        while True:
            numDisparities = cv2.getTrackbarPos("numDisparities", "disp") * 16
            blockSize = cv2.getTrackbarPos("blockSize", "disp") * 2 + 5
            preFilterType = cv2.getTrackbarPos("preFilterType", "disp")
            preFilterSize = cv2.getTrackbarPos("preFilterSize", "disp") * 2 + 5
            preFilterCap = cv2.getTrackbarPos("preFilterCap", "disp")
            textureThreshold = cv2.getTrackbarPos("textureThreshold", "disp")
            uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disp")
            speckleRange = cv2.getTrackbarPos("speckleRange", "disp")
            speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disp") * 2
            disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disp")
            minDisparity = cv2.getTrackbarPos("minDisparity", "disp")

            stereo = cv2.StereoBM_create()

            # Setting the updated parameters before computing disparity map
            stereo.setNumDisparities(numDisparities)
            stereo.setBlockSize(blockSize)
            stereo.setPreFilterType(preFilterType)
            stereo.setPreFilterSize(preFilterSize)
            stereo.setPreFilterCap(preFilterCap)
            stereo.setTextureThreshold(textureThreshold)
            stereo.setUniquenessRatio(uniquenessRatio)
            stereo.setSpeckleRange(speckleRange)
            stereo.setSpeckleWindowSize(speckleWindowSize)
            stereo.setDisp12MaxDiff(disp12MaxDiff)
            stereo.setMinDisparity(minDisparity)

            # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            disparity_image = stereo.compute(rectified_left, rectified_right).astype(
                np.float32
            )
            disparity_image = (disparity_image / 16.0 - minDisparity) / numDisparities
            # Displaying the disparity map
            cv2.imshow("disp", disparity_image)

            # button to save parameters
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                print("numDisparities: ", numDisparities)
                print("blockSize: ", blockSize)
                print("preFilterType: ", preFilterType)
                print("preFilterSize: ", preFilterSize)
                print("preFilterCap: ", preFilterCap)
                print("textureThreshold: ", textureThreshold)
                print("uniquenessRatio: ", uniquenessRatio)
                print("speckleRange: ", speckleRange)
                print("speckleWindowSize: ", speckleWindowSize)
                print("disp12MaxDiff: ", disp12MaxDiff)
                print("minDisparity: ", minDisparity)

            elif k == ord("c"):
                break

            # if cv2.waitKey(1) == 27:
            #     break

        return disparity_image


def stitch_result(left_right, disparity, stereo_depth):
    # disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)
    disparity = stereo_depth.map_to_color(disparity)
    pad_width = int(disparity.shape[1] / 2)
    disparity = np.pad(
        disparity,
        ((0, 0), (pad_width, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    combined_all = np.vstack((left_right, disparity))
    # print(combined_all.shape)
    return combined_all


def single_inference(stereo_depth, capture, method, frame_idx=None):
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx is None:
        frame_idx = np.random.randint(0, total_frames)
        print(frame_idx)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = capture.read()
    left, right = np.split(frame, 2, axis=1)
    disparity_image = stereo_depth.get_disparity(left, right, method=method)
    return frame, disparity_image


def stich_result_with_confidence(left_right, disparity, confidence, stereo_depth):
    # disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)
    # disparity = stereo_depth.map_to_color(disparity)
    confidence = cv2.cvtColor(confidence, cv2.COLOR_GRAY2BGR)
    disp_conf = np.hstack((disparity, confidence))
    combined_all = np.vstack((left_right, disp_conf))
    # print(combined_all.shape)
    return combined_all


def video_inference(stereo_depth, capture, method, output_path="output.avi"):
    out_size = stereo_depth.image_size[0] * 2, stereo_depth.image_size[1] * 2
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, 30, out_size)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="Processing Frames"):
        ret, frame = capture.read()
        if not ret:
            break
        # top, thermal = np.split(frame, 2, axis=0)
        left, right = np.split(frame, 2, axis=1)
        # print(left.shape, right.shape)
        disparity_image, confidence_img, frame_rectified = stereo_depth.get_disparity(
            left, right, method=method
        )
        # result = stitch_result(frame, disparity_image, stereo_depth)
        result = stich_result_with_confidence(
            frame_rectified, disparity_image, confidence_img, stereo_depth
        )
        out.write(result)


def vertical_interpolation(image, blank_color=[0, 0, 0]):
    # Load the image
    height, width, channels = image.shape

    # Create a mask for the blank spots
    mask = np.all(image == blank_color, axis=2)

    # Iterate through each channel and perform interpolation for the blank spots vertically (along columns)
    for i in range(channels):
        channel = image[:, :, i]
        channel = channel.astype(
            "float32"
        )  # Convert to float for accurate interpolation
        channel[mask] = np.nan  # Set blank spots to NaN for interpolation

        # Linear interpolation along the columns to fill in NaN values
        for col in range(width):
            nan_indices = np.isnan(channel[:, col])
            non_nan_indices = ~nan_indices
            channel[nan_indices, col] = np.interp(
                nan_indices.nonzero()[0],
                non_nan_indices.nonzero()[0],
                channel[non_nan_indices, col],
            )

        # Assign the interpolated channel back to the image
        image[:, :, i] = channel

    # Convert back to uint8
    image = image.astype("uint8")
    return image


def main():
    # args
    left_camera_yaml = "./thermal_left.yaml"
    right_camera_yaml = "./thermal_right.yaml"
    video_file = "/home/devansh/airlab/depth_estimation/vpi_scripts/2023-11-07-Thermal_Test_video.avi"

    # load calibration data
    left_camera_properties = CameraPropertiesLoader(left_camera_yaml)
    right_camera_properties = CameraPropertiesLoader(right_camera_yaml)

    stereo_depth = StereoDepthInference(
        left_camera_properties, right_camera_properties, image_size=(640, 512)
    )

    # load video
    capture = cv2.VideoCapture(video_file)

    video_inference(
        stereo_depth,
        capture,
        method="vpi",
        output_path="2023-11-07-Thermal_Test_result_rgb_conf500.avi",
    )

    # frame, disparity = single_inference(
    #     stereo_depth, capture, method="vpi", frame_idx=1137
    # )
    # result = stitch_result(frame, disparity, stereo_depth)
    # # show image
    # cv2.imshow("combined", result)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
