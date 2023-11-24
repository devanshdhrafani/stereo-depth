import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from tqdm import tqdm


class ROSBagImageExtractor:
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.bridge = CvBridge()

    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_images(self, topics=["/thermal_left/image", "/thermal_right/image"]):
        with rosbag.Bag(self.bag_file, "r") as bag:
            total_messages = bag.get_message_count()
            message_iterator = bag.read_messages(topics=topics)
            with tqdm(total=total_messages, desc="Processing Messages") as pbar:
                for topic, msg, t in message_iterator:
                    left_or_right = topic.split("/")[1]
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                    if cv_image.dtype == "uint16":
                        img_8 = cv2.normalize(
                            cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                        )
                    else:
                        img_8 = cv_image
                    image_timestamp = int(t.to_nsec() / 1e6)
                    if not os.path.exists(os.path.join(self.output_dir, left_or_right)):
                        os.makedirs(
                            os.path.join(self.output_dir, left_or_right), exist_ok=True
                        )
                    image_filename = os.path.join(
                        self.output_dir, left_or_right, f"{image_timestamp}.png"
                    )
                    cv2.imwrite(image_filename, img_8)
                    # print(f"Saved image: {image_filename}")
                    pbar.update(1)

        print("Image extraction complete")


if __name__ == "__main__":
    bag_file = (
        "/media/devansh/storage/wildfire/2023-11-07-Thermal_Test/thermal_2023-11-07.bag"
    )
    output_dir = "./2023-11-07-Thermal_Test/images"

    image_extractor = ROSBagImageExtractor(bag_file, output_dir)
    image_extractor.create_output_directory()
    image_extractor.extract_images()
