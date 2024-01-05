import cv2
import numpy as np
import matplotlib.pyplot as plt


class ThermalImageProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def process_image(self):
        # Load the 16-bit thermal image
        img_16 = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
        img_8_scaled = (img_16 / 255).astype(np.uint8)

        # outlier removal
        img_16_cleaned = img_16.copy()

        # find threshold using mean and std
        mean = np.mean(img_16_cleaned)
        std = np.std(img_16_cleaned)
        threshold = mean + 2 * std
        img_16_cleaned[img_16_cleaned > threshold] = mean

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

        # Display the original image in the first subplot
        ax1.imshow(cv2.cvtColor((img_16 / 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))
        ax1.set_title("Original 16-bit Image (scaled to 8-bit)")

        # Display the histogram plot in the second subplot
        ax2.hist(img_16.flatten(), bins=500)
        ax2.set_xlabel("Pixel Values")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Histogram of 16-bit Image")

        # Remove the space between subplots
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()

        # Save the figure
        # plt.savefig("hot_object_histogram.png")

        lower_bound = 21000
        upper_bound = 24000

        # lower_bound = np.percentile(img_16_cleaned, 1)
        # upper_bound = np.max(img_16_cleaned)

        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")

        # Normalize the image to 8-bit while emphasizing the pixels in the range [lower_bound, upper_bound]
        histogram_bound_img = np.zeros_like(img_16, dtype=np.uint8)
        mask = (img_16 >= lower_bound) & (img_16 <= upper_bound)
        histogram_bound_img[mask] = (
            (img_16[mask] - lower_bound) / (upper_bound - lower_bound) * 255
        ).astype(np.uint8)

        # fill values above upper_bound with mean value
        histogram_bound_img[img_16 > upper_bound] = np.mean(
            histogram_bound_img[img_16 <= upper_bound]
        )

        # Create a figure with 4 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))

        # Display the original image in the first subplot
        ax1.imshow(cv2.cvtColor(img_8_scaled, cv2.COLOR_GRAY2RGB))
        ax1.set_title("Original 16-bit Image (scaled to 8-bit)")

        # Display the enhanced image in the second subplot
        ax2.imshow(cv2.cvtColor(histogram_bound_img, cv2.COLOR_GRAY2RGB))
        ax2.set_title(f"Intensity bound [{lower_bound}, {upper_bound}] Image")

        minmax_image = cv2.normalize(img_16, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        ax3.imshow(cv2.cvtColor(minmax_image, cv2.COLOR_GRAY2RGB))
        ax3.set_title("Min-Max Normalized Image")

        histogram_minmax_img = np.zeros_like(img_16, dtype=np.uint16)
        mask = (img_16 >= lower_bound) & (img_16 <= upper_bound)
        histogram_minmax_img[mask] = (
            (img_16[mask] - lower_bound) / (upper_bound - lower_bound) * 65535
        ).astype(np.uint16)

        # fill values above upper_bound with mean value
        histogram_minmax_img[img_16 > upper_bound] = np.mean(
            histogram_minmax_img[img_16 <= upper_bound]
        )

        histogram_minmax_img = cv2.normalize(
            histogram_minmax_img, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        ax4.imshow(cv2.cvtColor(histogram_minmax_img, cv2.COLOR_GRAY2RGB))
        ax4.set_title("Intensity bound + Min-Max Normalized Image")

        # tight
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.0, hspace=0.2)

        # plt.savefig(output_path)
        plt.show()


if __name__ == "__main__":
    input_path = "/media/devansh/T7 Shield/wildfire_thermal/2.images/thermal_2023-11-07-throughTrees_trial2/images_raw/thermal_left/1677762119548.png"
    output_path = "./hot_object.png"

    img_16_processor = ThermalImageProcessor(input_path, output_path)
    img_16_processor.process_image()
