# examples/makemovie.py

import os
import cv2

# Parameters
image_folder = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'movie',
    '3d')
OUTPUT_VIDEO = 'output_video.mp4'
FPS = 1  # 20
COMPRESSION_RATIO = 1  # 0.5

# Get the image file names
# images = [f'{i}.png' for i in range(200, -1, -1)] + [f'{i}.png' for i in range(201)]
images = ['esaxs.png', 'saxs.png', 'maxs.png', 'waxs.png']

# Get the dimensions of the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Resize dimensions
new_width = int(width * COMPRESSION_RATIO)
new_height = int(height * COMPRESSION_RATIO)

# Set the video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the video writer
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (new_width, new_height))

# Write the frames to the video
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    frame_resized = cv2.resize(frame, (new_width, new_height))
    video.write(frame_resized)

# Release the video writer
video.release()

print(f"Video saved as {OUTPUT_VIDEO}")
