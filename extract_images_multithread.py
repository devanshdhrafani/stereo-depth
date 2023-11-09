import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class ROSBagImageExtractor:
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.bridge = CvBridge()

    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_image(self, topic, msg, t):
        left_or_right = topic.split('/')[1]
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image_timestamp = int(t.to_nsec() / 1e6)
        if not os.path.exists(os.path.join(self.output_dir, left_or_right)):
            os.makedirs(os.path.join(self.output_dir, left_or_right), exist_ok=True)
        image_filename = os.path.join(self.output_dir, left_or_right, f"{image_timestamp}.png")
        cv2.imwrite(image_filename, cv_image)

    def extract_images(self, topics=['/thermal_left/image', '/thermal_right/image']):
        with rosbag.Bag(self.bag_file, 'r') as bag:
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
    bag_file = '/media/devansh/storage/wildfire/2023-04-27-FIRE-SGL/wildfire_2023-03-02-07-58-54.bag'
    output_dir = '/media/devansh/storage/wildfire/wildfire_2023-03-02-07-58-54_images'

    image_extractor = ROSBagImageExtractor(bag_file, output_dir)
    image_extractor.create_output_directory()
    image_extractor.extract_images()
