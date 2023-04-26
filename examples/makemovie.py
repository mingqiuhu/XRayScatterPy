import cv2
import os

# Parameters
image_folder = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'movie',
    'real_space')
output_video = 'output_video.mp4'
fps = 1  # 20
compression_ratio = 1  # 0.5

# Get the image file names
# images = [f'{i}.png' for i in range(200, -1, -1)] + [f'{i}.png' for i in range(201)]
images = ['esaxs.png', 'saxs.png', 'maxs.png', 'waxs.png']

# Get the dimensions of the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Resize dimensions
new_width = int(width * compression_ratio)
new_height = int(height * compression_ratio)

# Set the video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create the video writer
video = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

# Write the frames to the video
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    frame_resized = cv2.resize(frame, (new_width, new_height))
    video.write(frame_resized)

# Release the video writer
video.release()

print(f"Video saved as {output_video}")
