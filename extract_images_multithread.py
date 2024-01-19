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
        self.raw = raw

    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def create_metadata(self):
        metadata_filename = os.path.join(self.output_dir, "extracted_metadata.txt")
        print(f"Saving metadata to: {metadata_filename}")
        with open(metadata_filename, "w") as f:
            f.write(f"Bag file: {self.bag_file}\n")
            f.write(f"Raw: {self.raw}\n")

    def update_metadata(self, msg):
        with open(os.path.join(self.output_dir, "extracted_metadata.txt"), "a") as f:
            f.write(f"Encoding: {msg.encoding}\n")
            f.write(f"Height: {msg.height}\n")
            f.write(f"Width: {msg.width}\n")

    def extract_image(self, topic, msg, t):
        left_or_right = topic.split("/")[1]
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        img = cv_image
        image_timestamp = int(t.to_nsec() / 1e6)
        if not os.path.exists(os.path.join(self.output_dir, left_or_right)):
            os.makedirs(os.path.join(self.output_dir, left_or_right), exist_ok=True)
        image_filename = os.path.join(
            self.output_dir, left_or_right, f"{image_timestamp}.png"
        )
        cv2.imwrite(image_filename, img)

    def extract_images(
        self, multi_thread=True, topics=["/thermal_left/image", "/thermal_right/image"]
    ):
        with rosbag.Bag(self.bag_file, "r") as bag:
            total_messages = 0
            for topic in topics:
                total_messages += bag.get_message_count(topic)
            message_iterator = bag.read_messages(topics=topics)
            with tqdm(total=total_messages, desc="Processing Messages") as pbar:
                if multi_thread == False:
                    for topic, msg, t in message_iterator:
                        if pbar.n == 0:
                            self.update_metadata(msg)
                        self.extract_image(topic, msg, t)
                        pbar.update(1)
                else:
                    with ThreadPoolExecutor() as executor:
                        for topic, msg, t in message_iterator:
                            if pbar.n == 0:
                                self.update_metadata(msg)
                            executor.submit(self.extract_image, topic, msg, t)
                            pbar.update(1)

        print("Image extraction complete")


if __name__ == "__main__":
    bag_file = (
        "/media/devansh/T7 Shield/wildfire_thermal/1.bags/fire_sgl_228/flight_2.bag"
    )
    output_dir = (
        "/media/devansh/T7 Shield/wildfire_thermal/2.images/fire_sgl_2/images_raw"
    )

    image_extractor = ROSBagImageExtractor(bag_file, output_dir, raw=True)
    image_extractor.create_output_directory()
    image_extractor.create_metadata()
    image_extractor.extract_images(multi_thread=False)
