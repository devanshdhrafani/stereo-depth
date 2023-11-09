import os
import cv2
import numpy as np
import tqdm

class ImageSync:
    def __init__(self, left_folder, right_folder, output_video, tolerance):
        self.left_folder = left_folder
        self.right_folder = right_folder
        self.output_video = output_video
        self.tolerance = tolerance

    def names_to_timestamp(self, names):
        timestamps = []
        for name in names:
            timestamps.append(int(name.split(".")[0]))
        timestamps = np.array(timestamps)
        return timestamps
    
    def get_closest_timestamp(self, timestamps, target):
        closest_idx = np.argmin(np.abs(timestamps - target))
        return timestamps[closest_idx], np.abs(timestamps[closest_idx]-target)

    def sync_images(self):
        left_images = sorted(os.listdir(self.left_folder))
        right_images = sorted(os.listdir(self.right_folder))

        left_timestamps = self.names_to_timestamp(left_images)
        right_timestamps = self.names_to_timestamp(right_images)

        synced_images = []

        for left_image in left_images:
            left_timestamp = int(left_image.split(".")[0])
            closest_right_timestamp, error = self.get_closest_timestamp(right_timestamps, left_timestamp)

            if error < self.tolerance:
                synced_images.append((left_image, f"{closest_right_timestamp}.png"))

        return synced_images

    def create_video(self):
        synced_images = self.sync_images()
        video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 512))

        for left_image, right_image in tqdm.tqdm(synced_images, total=len(synced_images)):
            left_path = os.path.join(self.left_folder, left_image)
            right_path = os.path.join(self.right_folder, right_image)
            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)
            try:
                side_by_side = cv2.hconcat([left_image, right_image])
            except Exception as e:
                print(f"Encountered error: {e} at {left_image} and {right_image}. Skipping...")
                continue
            video_writer.write(side_by_side)

        video_writer.release()

if __name__ == '__main__':
    left_folder = "/media/devansh/storage/wildfire/wildfire_2023-03-02-07-58-54_images/thermal_left"
    right_folder = "/media/devansh/storage/wildfire/wildfire_2023-03-02-07-58-54_images/thermal_right"
    output_video = "wildfire_2023-03-02-07-58-54_video.avi"
    tolerance = 500 # ms

    image_sync = ImageSync(left_folder, right_folder, output_video, tolerance)
    image_sync.create_video()