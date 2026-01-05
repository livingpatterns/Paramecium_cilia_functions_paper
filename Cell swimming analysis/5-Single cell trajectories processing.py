"""
# Extraction and processing of single cell trajectories
This script extracts, processes, and visualizes single cell swimming trajectories from filtered trajectory arrays.
It handles short gaps of missing data (NaNs) by interpolation, segments each trajectory into continuous parts, and saves or visualizes the longest valid segment for each cell.

Input
- path: Directory containing the `filtered_trajectories.npy` file.
- filtered_trajectories.npy: Numpy array of shape (n_trajectories, n_frames, 2) with (x, y) positions, possibly containing NaNs.
- px_mm: Conversion factor from pixels to millimeters.
- max_nan_length: Maximum length of consecutive NaNs to interpolate.
- min_segment_length: Minimum length of continuous segments to consider for saving/plotting.

Output:
- For each trajectory: longest continuous segment saved as `.npy` in pixels and mm.
- Plots of each trajectory's longest segment colored by time (seconds).
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# INPUT PARAMETERS
path = 'W:/Users/Daphne/Imaging_Daphne/25-09-08_RPi_ptetwt_swimming/bgd_subs_4000/processed_080925_ptet_swimming11_analysis/'
file = 'filtered_trajectories.npy'
px_mm = 91.4  # pixels per mm
max_nan_length = 10
min_segment_length = 1800  # frames (1 min at 30 fps)


# Funtion definitions
def interpolate_short_nan_sequences(data, max_nan_length):
    """
    Interpolates sequences of NaN rows if their length is less than max_nan_length.
    """
    not_nan_mask = ~np.isnan(data).any(axis=1) # identify rows without NaN values
    change_indices = np.where(np.diff(not_nan_mask.astype(int)) != 0)[0] + 1 # Find indices where the mask changes
    segments_indices = np.concatenate(([0], change_indices, [len(data)])) # Add the start and end indices to define segments
    for i in range(len(segments_indices) - 1):
        start, end = segments_indices[i], segments_indices[i + 1]
        if np.isnan(data[start:end]).all() and (end - start) < max_nan_length: # Check if the current segment contains NaN and is short enough
            # Linearly interpolate between the previous and next non-NaN rows
            if start > 0 and end < len(data): # Ensure interpolation is possible
                prev_row = data[start - 1]
                next_row = data[end]
                interpolated_rows = np.linspace(prev_row, next_row, end - start + 2, axis=0)[1:-1]
                data[start:end] = interpolated_rows
    return data

def find_longest_segment(data):
    not_nan_mask = ~np.isnan(data).any(axis=1) # Identify rows without NaN after interpolation
    change_indices = np.where(np.diff(not_nan_mask.astype(int)) != 0)[0] + 1 # Find indices where the mask changes
    segments_indices = np.concatenate(([0], change_indices, [len(data)])) # Add the start and end indices to segment properly
    # Extract segments without NaN values
    segments = [
        data[segments_indices[i]:segments_indices[i+1]]
        for i in range(len(segments_indices) - 1)
        if not np.isnan(data[segments_indices[i]:segments_indices[i+1]]).any()
    ]
    if segments:
        return max(segments, key=len)
    else:
        return None

def save_trajectory(segment, path, file_prefix, trajectory_number, px_mm):
    np.save(os.path.join(path, f"{file_prefix}_{trajectory_number}.npy"), segment)
    np.save(os.path.join(path, f"{file_prefix}_{trajectory_number}_mm.npy"), segment / px_mm)

def plot_trajectory(segment, px_mm, path, file_prefix, trajectory_number):
    x = segment[:, 0] / px_mm
    y = segment[:, 1] / px_mm
    time_sec = np.arange(segment.shape[0]) / 30
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, c=time_sec, cmap="magma", s=1)
    plt.colorbar(label='time (sec)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(f'Trajectory {trajectory_number} - {file_prefix}')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(path, f"{file_prefix}_trajectory_{trajectory_number}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Main processing loop: loop through each trajectory_number in track_array and interpolate the NaN values when less than 10 consecutive NaN values
track_array = np.load(os.path.join(path, file), allow_pickle=True)
for trajectory_number in range(track_array.shape[0]):
    track_data = track_array[trajectory_number, :, :]  # Extract the trajectory data for the current trajectory_number
    data = interpolate_short_nan_sequences(track_data, max_nan_length=max_nan_length) # Interpolate short NaN sequences
    longest_segment = find_longest_segment(data)
    if longest_segment is not None and len(longest_segment) >= min_segment_length:
        save_trajectory(longest_segment, path, file[:-4], trajectory_number, px_mm) # Save the longest segment as a .npy file with the trajectory number in the name
        plot_trajectory(longest_segment, px_mm, path, file[:-4], trajectory_number) # Plot the trajectory of the longest segment
        print(f"Trajectory {trajectory_number}: saved, length = {len(longest_segment)}")
    else:
        print(f"Trajectory {trajectory_number}: no valid segment found or too short.")