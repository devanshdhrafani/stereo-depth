import os
import cv2
import numpy as np
import tqdm
from rectify_image import RectifyImage


class ImageSync:
    def __init__(self, left_folder, right_folder, output_video, tolerance):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.output_video = output_video
        self.tolerance = tolerance
        self.rectify = RectifyImage("./thermal_left.yaml", "./thermal_right.yaml")

    def names_to_timestamp(self, names):
        timestamps = []
        for name in names:
            timestamps.append(int(name.split(".")[0]))
        timestamps = np.array(timestamps)
        return timestamps

    def get_closest_timestamp(self, timestamps, target):
        closest_idx = np.argmin(np.abs(timestamps - target))
        return timestamps[closest_idx], np.abs(timestamps[closest_idx] - target)

    def sync_images(self):
        left_images = sorted(os.listdir(self.left_folder))
        right_images = sorted(os.listdir(self.right_folder))

        print(f"Found {len(left_images)} left images")
        print(f"Found {len(right_images)} right images")

        left_timestamps = self.names_to_timestamp(left_images)
        right_timestamps = self.names_to_timestamp(right_images)

        synced_images = []

        for left_image in left_images:
            left_timestamp = int(left_image.split(".")[0])
            closest_right_timestamp, error = self.get_closest_timestamp(
                right_timestamps, left_timestamp
            )

            if error < self.tolerance:
                synced_images.append((left_image, f"{closest_right_timestamp}.png"))

        return synced_images

    def process_image(self, img_16, type=None, outlier_removal=True):
        if outlier_removal:
            # upper bound is 4 std above mean
            img_16_cleaned = img_16.copy()
            # mean = np.mean(img_16_cleaned)
            # std = np.std(img_16_cleaned)
            # upper_threshold = mean + 4 * std
            # lower_threshold = mean - 4 * std

            # print(f"Upper threshold: {upper_threshold}")
            # print(f"Lower threshold: {lower_threshold}")

            upper_threshold = 27000
            lower_threshold = 20000

            img_16_cleaned[img_16_cleaned > upper_threshold] = upper_threshold
            img_16_cleaned[img_16_cleaned < lower_threshold] = lower_threshold

            img_16 = img_16_cleaned

        if type == "histogram":
            lower_bound = 21000
            upper_bound = 24000
            # find the 99th percentile
            # lower_bound = np.percentile(img_16, 1)
            # upper_bound = np.max(img_16)

            # print(f"Lower bound: {lower_bound}")
            # print(f"Upper bound: {upper_bound}")

            histogram_bound_img = np.zeros_like(img_16, dtype=np.uint8)
            mask = (img_16 >= lower_bound) & (img_16 <= upper_bound)
            histogram_bound_img[mask] = (
                (img_16[mask] - lower_bound) / (upper_bound - lower_bound) * 255
            ).astype(np.uint8)

            histogram_bound_img = cv2.normalize(
                histogram_bound_img, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            histogram_bound_img = cv2.cvtColor(histogram_bound_img, cv2.COLOR_GRAY2RGB)
            return histogram_bound_img
        elif type == "minmax":
            minmax_image = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            minmax_image = cv2.cvtColor(minmax_image, cv2.COLOR_GRAY2RGB)
            return minmax_image
        else:
            img_8 = (img_16 / 255).astype(np.uint8)
            img_8 = cv2.cvtColor(img_8, cv2.COLOR_GRAY2RGB)
            return img_8

    def create_video(self):
        synced_images = self.sync_images()

        # take the first image to get the shape
        left_path = os.path.join(self.left_folder, synced_images[0][0])
        left_image = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)

        video_shape = (left_image.shape[1] * 2, left_image.shape[0])

        video_writer = cv2.VideoWriter(
            self.output_video, cv2.VideoWriter_fourcc(*"MJPG"), 30, video_shape
        )

        for left_image, right_image in tqdm.tqdm(
            synced_images, total=len(synced_images)
        ):
            left_path = os.path.join(self.left_folder, left_image)
            right_path = os.path.join(self.right_folder, right_image)
            left_image = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
            right_image = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)

            left_image = self.process_image(
                left_image, type="histogram", outlier_removal=True
            )
            right_image = self.process_image(
                right_image, type="histogram", outlier_removal=True
            )

            try:
                side_by_side = cv2.hconcat([left_image, right_image])
            except Exception as e:
                print(
                    f"Encountered error: {e} at {left_image} and {right_image}. Skipping..."
                )
                continue
            video_writer.write(side_by_side)

        video_writer.release()

    def create_images(self):
        synced_images = self.sync_images()
        print(f"Found {len(synced_images)} synced images")

        # move two up from the left folder and create a new folder called images_raw_synced
        output_path = os.path.join(
            self.left_folder, "../../images_rectified_minmax_synced"
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f"Saving images to: {output_path}")

        frame_count = 0
        for left_image, right_image in tqdm.tqdm(
            synced_images, total=len(synced_images)
        ):
            left_path = os.path.join(self.left_folder, left_image)
            right_path = os.path.join(self.right_folder, right_image)
            left_image = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
            right_image = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)

            left_image, right_image, *_ = self.rectify(left_image, right_image)

            try:
                side_by_side = cv2.hconcat([left_image, right_image])
            except Exception as e:
                print(
                    f"Encountered error: {e} at {left_image} and {right_image}. Skipping..."
                )
                continue

            image_name = f"{frame_count}.png"
            image_path = os.path.join(output_path, image_name)
            cv2.imwrite(image_path, side_by_side)
            frame_count += 1


if __name__ == "__main__":
    left_folder = "/media/devansh/T7 Shield/wildfire_thermal/2.images/gascola_1/images_raw/thermal_left"
    right_folder = "/media/devansh/T7 Shield/wildfire_thermal/2.images/gascola_1/images_raw/thermal_right"
    output_video = "/media/devansh/T7 Shield/wildfire_thermal/3.synced_videos/2023-11-07-throughTrees_trial2_histogram.avi"
    tolerance = 1000  # ms

    image_sync = ImageSync(left_folder, right_folder, output_video, tolerance)
    # image_sync.create_video()
    image_sync.create_images()
