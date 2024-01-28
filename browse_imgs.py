import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="Path to the folder containing images")
parser.add_argument(
    "--skip",
    type=int,
    default=20,
    help="Number of frames to jump when pressing 'q' or 'e'",
)


def display_images(folder_path, frame_jump):
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()

    current_index = 0

    while True:
        image_path = os.path.join(folder_path, image_files[current_index])
        image = cv2.imread(image_path)

        # Add text to the top left corner
        image_name = os.path.basename(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 255)  # Yellow color
        font_thickness = 1
        cv2.putText(
            image,
            f"Idx {current_index}: {image_name}",
            (10, 20),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

        # Display the resized image
        cv2.imshow("Image Viewer", image)

        key = cv2.waitKey(0)

        if key == 27:  # Press 'Esc' to exit
            break
        elif key == 81 or key == 63234:  # Left arrow key
            current_index = (current_index - 1) % len(image_files)
        elif key == 83 or key == 63235:  # Right arrow key
            current_index = (current_index + 1) % len(image_files)
        elif key == ord("q"):  # 'q' key to go backward by 10 frames
            current_index = (current_index - frame_jump) % len(image_files)
        elif key == ord("e"):  # 'e' key to go forward by 10 frames
            current_index = (current_index + frame_jump) % len(image_files)
        elif key == ord("s"):  # 's' key to print image name on terminal
            print(f"Current Image Name: {image_name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()

    display_images(args.dir, args.skip)
