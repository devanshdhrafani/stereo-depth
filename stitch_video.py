import cv2

# Define the file paths for the input videos
video1_path = 'thermal_left_1677762089_183333337.mp4'
video2_path = 'thermal_right_1677762089_0.mp4'

# Open the video files
video1 = cv2.VideoCapture(video1_path)
video2 = cv2.VideoCapture(video2_path)

# Check if the videos were opened successfully
if not video1.isOpened() or not video2.isOpened():
    print("Error: Could not open video files.")
    exit()

# Get the properties of the input videos
frame_width = int(video1.get(3)) + int(video2.get(3))
frame_height = int(video1.get(4))

# Create an output video writer
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

while True:
    # Read frames from the input videos
    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()

    # Break the loop if either of the videos is finished
    if not ret1 or not ret2:
        break

    # Resize the frames to have the same height (adjust width accordingly)
    frame1 = cv2.resize(frame1, (int(frame_height * frame1.shape[1] / frame1.shape[0]), frame_height))
    frame2 = cv2.resize(frame2, (int(frame_height * frame2.shape[1] / frame2.shape[0]), frame_height))

    # Combine the frames side by side
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write the combined frame to the output video
    out.write(combined_frame)

# Release the video objects and writer
video1.release()
video2.release()
out.release()

# Print a message indicating the process is complete
print("Videos are successfully stitched together and saved as 'output.mp4'")
