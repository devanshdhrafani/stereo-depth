
import cv2

# Open the video file
video = cv2.VideoCapture('result_02.avi')

# Get the video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('result_02_thermal.avi', fourcc, fps, (width, height//2))

# Loop through the frames and extract the bottom half
while True:
    ret, frame = video.read()
    if not ret:
        break
    bottom_half = frame[height//2:height, :]
    out.write(bottom_half)

# Release the video objects
video.release()
out.release()
cv2.destroyAllWindows()
