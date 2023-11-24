import cv2
import numpy as np
import matplotlib.pyplot as plt


class ThermalImageProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def process_image(self):
        # Load the 16-bit thermal image
        thermal_image = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
        thermal_image_backup = thermal_image.copy()

        # # Plot the histogram of the 16-bit thermal image
        # plt.hist(thermal_image.flatten(), bins=100)
        # # label the axes
        # plt.xlabel("Pixel Values")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of 16-bit Thermal Image")
        # plt.show()

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image in the first subplot
        ax1.imshow(thermal_image, cmap="gray")
        ax1.set_title("Original Thermal Image")

        # Display the histogram plot in the second subplot
        ax2.hist(thermal_image.flatten(), bins=100)
        ax2.set_xlabel("Pixel Values")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Histogram of 16-bit Thermal Image")

        # Remove the space between subplots
        plt.subplots_adjust(wspace=0.3)

        # Save the figure
        plt.savefig("histogram.png")

        lower_bound = 21000
        upper_bound = 24000

        # normalize the image to 8-bit while emaphasizing the pixels in the range
        # [lower_bound, upper_bound]
        # Normalize the image to 8-bit while emphasizing the pixels in the range [lower_bound, upper_bound]
        normalized_image = np.zeros_like(thermal_image, dtype=np.uint8)
        mask = (thermal_image >= lower_bound) & (thermal_image <= upper_bound)
        normalized_image[mask] = (
            (thermal_image[mask] - lower_bound) / (upper_bound - lower_bound) * 255
        ).astype(np.uint8)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image in the first subplot
        ax1.imshow(thermal_image, cmap="gray")
        ax1.set_title("Original Thermal Image")

        # Display the enhanced image in the second subplot
        ax2.imshow(normalized_image, cmap="gray")
        ax2.set_title("Enhanced Thermal Image")

        # Remove the space between subplots
        plt.subplots_adjust(wspace=0.3)

        # Save the figure
        plt.savefig("enhanced_image.png")

        # Apply all cv2 normalize functions to the image. Save the results to a single stiched image. Label each image with its name.
        normalized_image_cv2_1 = cv2.normalize(
            thermal_image, None, 0, 255, cv2.NORM_MINMAX
        )
        normalized_image_cv2_2 = cv2.normalize(
            thermal_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        normalized_image_cv2_3 = cv2.normalize(
            thermal_image, None, alpha=0, beta=255, norm_type=cv2.NORM_INF
        )
        normalized_image_cv2_4 = cv2.normalize(
            thermal_image, None, alpha=0, beta=255, norm_type=cv2.NORM_L1
        )
        normalized_image_cv2_5 = cv2.normalize(
            thermal_image, None, alpha=0, beta=255, norm_type=cv2.NORM_L2
        )

        thermal_image_original = thermal_image / 255

        # Concatenate all images into a single image
        concatenated_image = np.concatenate(
            (
                thermal_image_original,
                normalized_image_cv2_1,
                normalized_image_cv2_2,
                normalized_image_cv2_3,
                normalized_image_cv2_4,
                normalized_image_cv2_5,
            ),
            axis=1,
        )

        # Save the concatenated image
        cv2.imwrite("concatenated_images.png", concatenated_image)


if __name__ == "__main__":
    input_path = "2023-11-07-Thermal_Test/images_raw/thermal_left/1677762469779377.png"
    output_path = "./1677762469779377.png"

    thermal_image_processor = ThermalImageProcessor(input_path, output_path)
    thermal_image_processor.process_image()
