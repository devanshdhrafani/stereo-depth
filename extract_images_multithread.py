import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class ROSBagImageExtractor:
    def __init__(self, bag_file, output_dir, hist_equalize=False, raw=False):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.bridge = CvBridge()
        self.hist_equalize = hist_equalize
        self.raw = raw

    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_image(self, topic, msg, t):
        left_or_right = topic.split("/")[1]
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        if cv_image.dtype == "uint16" and not self.raw:
            img = cv2.normalize(
                cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        else:
            img = cv_image
        if self.hist_equalize:
            img = cv2.equalizeHist(img)
        image_timestamp = int(t.to_nsec() / 1e3)
        if not os.path.exists(os.path.join(self.output_dir, left_or_right)):
            os.makedirs(os.path.join(self.output_dir, left_or_right), exist_ok=True)
        image_filename = os.path.join(
            self.output_dir, left_or_right, f"{image_timestamp}.png"
        )
        # check if image already exists
        if os.path.exists(image_filename):
            print(f"Image already exists: {image_filename}")
        cv2.imwrite(image_filename, img)

    def extract_images(self, topics=["/thermal_left/image", "/thermal_right/image"]):
        with rosbag.Bag(self.bag_file, "r") as bag:
            total_messages = 0
            for topic in topics:
                total_messages += bag.get_message_count(topic)
            message_iterator = bag.read_messages(topics=topics)
            with tqdm(total=total_messages, desc="Processing Messages") as pbar:
                with ThreadPoolExecutor() as executor:
                    for topic, msg, t in message_iterator:
                        executor.submit(self.extract_image, topic, msg, t)
                        pbar.update(1)

        print("Image extraction complete")


if __name__ == "__main__":
    bag_file = (
        "/media/devansh/storage/wildfire/2023-11-07-Thermal_Test/thermal_2023-11-07.bag"
    )
    output_dir = "./2023-11-07-Thermal_Test/images_raw"

    image_extractor = ROSBagImageExtractor(bag_file, output_dir, raw=True)
    image_extractor.create_output_directory()
    image_extractor.extract_images()
