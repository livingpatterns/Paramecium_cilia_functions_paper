import numpy as np
import pims
import multiprocessing
import matplotlib.pyplot as plt
import os
import trackpy as tp


# Use pims pipeline to convert image to grayscale, taking only one color channel
@pims.pipeline
def gray(image):
    return image[:, :, 0].copy()  # Ensure a copy is returned to prevent unintended modifications

#this function deals with quality control fo particle identification:
def quality_control1(features, path, n_bins=10):
    """
    Plots the probability distribution of the number of particles found per frame

    Parameters:
    features (dataframe): Data frame with the list of positions of particles found in each frame.
    path (str): Path where the histogram will be saved.
    n_bins (int): Number of bins for the histogram. The bin width is considered 1 since particles are discrete

    Returns:
    nothing, only saves the histogram as a png image in the specified path.
    """
    savefolder=os.path.join(path,'plots')
    os.makedirs(savefolder, exist_ok=True)

    max_frame = features['frame'].max()
    num_part_frame = np.zeros(max_frame + 1)
    for k in range(len(num_part_frame)):
        num_part_frame[k] = features['frame'].eq(k).sum()

    plt.plot(num_part_frame)
    plt.savefig(os.path.join(savefolder, 'n_particles_frame.png'))
    plt.close()
    counts, bin_edges = np.histogram(num_part_frame, bins=np.arange(0, n_bins, 1))
    plt.bar(bin_edges[0:-1] - 0.5, counts / len(num_part_frame), align='edge', width=1)
    plt.xticks(bin_edges, bin_edges.astype(str))
    plt.xlabel('Number of particles per frame')
    plt.savefig(os.path.join(savefolder,'n_particles_found.png'))
    plt.close()




# Use 'spawn' start method for multiprocessing on Windows, 'fork' for mac and Linux
if os.name == 'nt':
    multiprocessing.set_start_method('spawn', force=True)
else:
    multiprocessing.set_start_method('fork', force=True)

# Specify the path with your microscopy images
path = 'W:/Users/Daphne/Imaging_Daphne/25-11-18_RPi_ptetwt_swimming_deciliated/bgd_subs_4000/'

# Debugging print statements
print("Checking path:", path)
print("Path exists:", os.path.exists(path))

# List all videos in the path sorted by name
video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')])
print("Number of video files found:", len(video_files))

# Ensure at least one video exists
if not video_files:
    raise FileNotFoundError("No .mp4 video files found in the specified directory!")

n_video = 0  # Video to be analyzed for testing

# Ensure valid video selection
if n_video >= len(video_files):
    raise IndexError(f"n_video={n_video} is out of range! Max index: {len(video_files) - 1}")


# Open video safely
video = pims.open(video_files[n_video])
try:
    video = gray(video)  # Convert to grayscale
    n_frame = 500  # Frame to be analyzed for testing

    # Ensure the frame index is within range
    if n_frame >= len(video):
        raise ValueError(f"Frame index {n_frame} is out of range. Max index: {len(video) - 1}")

    # Particle tracking parameters: change for other cell types / conditions
    diameter = 17
    minmass = 800

    # Extract and process the frame
    test_frame = video[n_frame].copy()
    test = tp.locate(test_frame, diameter=diameter, minmass=minmass)

    # To try for brightfield / phase contrast videos
    # test = tp.locate_brightfield_ring(test_frame, diameter=diameter)

    # Annotate and display results
    tp.annotate(test, test_frame)
    plt.imshow(test_frame, cmap="gray")
    plt.show()

finally:
    del video  # Ensure cleanup of PIMS video object

# Process all videos in the directory
for n_video in range(len(video_files)):
    video_name = os.path.basename(video_files[n_video])  # Extract video filename
    analysis_path = os.path.join(os.path.dirname(path), video_name[:-4] + '_analysis')  # Create analysis folder
    os.makedirs(analysis_path, exist_ok=True)  # Ensure folder exists

    video = pims.open(video_files[n_video])
    try:
        video = gray(video)  # Convert to grayscale

        # Convert frames to a list before processing
        frames = [frame.copy() for frame in video]

        # Run TrackPy particle tracking
        tp.quiet()  # Suppress progress output
        particles = tp.batch(frames, diameter=diameter, minmass=minmass, processes=1)  # Debug with 1 process

        # Sanity check for particle identification
        quality_control1(particles, analysis_path, n_bins=10)

        # Save results
        particles.to_pickle(os.path.join(analysis_path, 'all_raw_particle_positions.pkl'))

        # Save parameters used
        with open(os.path.join(analysis_path, 'parameters_part_id.txt'), 'w') as f:
            f.write(f'diameter={diameter}\n')
            f.write(f'minmass={minmass}\n')

    finally:
        del video  # Ensure cleanup of PIMS video object

print("Processing complete!")

