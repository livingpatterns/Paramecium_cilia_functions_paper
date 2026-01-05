"""
# Background subtraction for cell swimming video analysis
This script performs background subtraction on cell swimming videos to enhance tracking of moving cells in later steps of the analysis.
It processes all `.mp4` videos in a specified directory by dividing each video into chunks, calculating a median background for each chunk, and subtracting it from the frames.
The processed videos are saved for further analysis.

Input:
- VIDEO_PATH: Directory containing input `.mp4` videos
- CHUNK_SIZE: Number of frames per chunk for background calculation
- MAX_WORKERS: Number of parallel threads for processing
- MACRO_BLOCK_SIZE: Ensures frame dimensions are compatible with video encoding

Output:
- Preprocessed `.mp4` videos with background subtracted, saved in a subdirectory of VIDEO_PATH
- A text file with processing parameters used (`parameters.txt`)
"""

import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import imageio

# Define constants
CHUNK_SIZE = 500 # increase when video contains very slow moving cells
MAX_WORKERS = 2
MACRO_BLOCK_SIZE = 1
VIDEO_PATH = 'W:/Users/Daphne/Imaging_Daphne/25-11-18_RPi_ptetwt_swimming_deciliated/'

SAVE_PATH = os.path.join(VIDEO_PATH, 'bgd_subs_' + str(CHUNK_SIZE))
os.makedirs(SAVE_PATH, exist_ok=True)  # Create folder if it doesn't exist

# Function to resize frame to dimensions divisible by macro_block_size
def resize_frame(frame, macro_block_size=1):
    height, width = frame.shape[:2]
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized_frame, dtype=np.uint8)

# Function to read video frame-by-frame (lazy loading)
def read_video_frames(video_path, macro_block_size=1):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale, resize, and ensure memory alignment
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        yield resize_frame(gray_frame, macro_block_size)
    cap.release()

# Function to compute median background in chunks
def calculate_background(frames):
    frames_list = list(frames)
    return np.median(frames_list, axis=0).astype(np.uint8)

# Function to process a single frame
def process_frame(frame, background):
    processed_frame = np.uint8(np.abs(frame.astype(float) - background.astype(float)))
    return np.ascontiguousarray(processed_frame, dtype=np.uint8)

# Process video file
def process_video(video_file):
    start_time = time.time()  # Start timing
    video_path = os.path.join(VIDEO_PATH, video_file)
    frames = read_video_frames(video_path, MACRO_BLOCK_SIZE)

    # Compute number of chunks
    frame_list = list(frames)  # Load frames lazily
    num_chunks = (len(frame_list) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Compute background frames in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunks = [frame_list[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
        bgds = list(executor.map(calculate_background, chunks))

    # Process frames and write output
    output_path = os.path.join(SAVE_PATH, "processed_" + video_file)
    with imageio.get_writer(output_path, fps=30, macro_block_size=1) as writer:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for chunk, bgd in zip(chunks, bgds):
                processed_chunk = list(executor.map(lambda frame: process_frame(frame, bgd), chunk))
                for frame in processed_chunk:
                    writer.append_data(frame)

    end_time = time.time()  # End timing
    print(f"Processing time for {video_file}: {end_time - start_time:.2f} seconds")


# Process all videos in the directory
start_time_all = time.time()  # Start timing for all videos
video_files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(process_video, video_files)

end_time_all = time.time()  # End timing for all videos
print(f"Total execution time: {end_time_all - start_time_all:.2f} seconds")

# Write a text file with the parameters used to process the video
file_name = os.path.join(SAVE_PATH, 'parameters.txt')
with open(file_name, 'w') as f:
    f.write(f'CHUNK_SIZE = {CHUNK_SIZE}\n')
    f.write(f'MAX_WORKERS = {MAX_WORKERS}\n')
    f.write(f'MACRO_BLOCK_SIZE = {MACRO_BLOCK_SIZE}\n')
    f.write(f'VIDEO_PATH = {VIDEO_PATH}\n')
    f.write(f'VIDEO_FILES = {video_files}\n')
    f.close()