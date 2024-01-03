import cv2
import os
import tqdm


class VideoToFramesConverter:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path

    def convert_to_frames(self):
        # Open the video file
        video = cv2.VideoCapture(self.video_path)

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read and save each frame as a PNG image
        frame_count = 0
        for frame_count in tqdm.tqdm(range(total_frames)):
            # Read the next frame
            ret, frame = video.read()

            # Break the loop if no more frames are available
            if not ret:
                break

            # Save the frame as a PNG image
            frame_path = os.path.join(self.output_path, f"{frame_count}.png")
            cv2.imwrite(frame_path, frame)

            frame_count += 1

        # Release the video file
        video.release()


# Usage example
video_path = "/media/devansh/T7 Shield/wildfire_thermal/3.synced_videos/2023-04-27_fireSGL_minmax.avi"
output_path = "/media/devansh/T7 Shield/wildfire_thermal/2.images/thermal_2023-04-27_fireSGL/minmax_synced"
converter = VideoToFramesConverter(video_path, output_path)
converter.convert_to_frames()
