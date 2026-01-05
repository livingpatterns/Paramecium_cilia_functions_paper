"""
# Trajectory filtering and visualization for cell swimming video analysis
This script processes and filters tracked cell trajectories from multiple analysis folders.
It loads trajectory data, calculates and visualizes average particle speeds, filters out trajectories below a speed threshold (to remove noise), and provides tools for visualizing and manually correcting overlapping or switching trajectories.
Filtered trajectories are saved for further analysis.
It is possible to process all folders at once or one folder at a time for manual inspection and correction.

Input:
- path: Directory containing analysis folders with tracked particle trajectories (`all_tracks_tp.pkl`)
- speed_threshold: Minimum average speed (in pixels/sec) for a trajectory to be retained (depends on cell condition and video pixel size)
- Optional manual correction: Options to merge or remove trajectories by editing the code
- Parameters inside function definitions can be adjusted, see comments per function for details.

Output:
- Filtered trajectories saved as `.npy` files in each analysis folder
- Plots of average speeds and time-color trajectories for visual inspection
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = 'W:/Users/Daphne/Imaging_Daphne/25-12-19_RPi_ptetwt_swimming_deciliated/crop/' # specify the path where your videos are saved

# Get folders that don't contain '.' in their name
folders = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
# Get .pkl files grouped by folder
files = {folder: [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('all_tracks_tp.pkl')] for folder in folders}

#These following functions deal with importing the trajectories, visualizing them and sorting them for analysis:
def get_tracks(track_files):
    "This function reads the tracking data from the .pkl files and organizes it into numpy arrays."
    # Read tracking data from .pkl files
    tracks = [pd.read_pickle(f) for f in track_files]

    # Find out many tracks are in total from all files and what is the maximum frame number
    n_tracks = sum(len(track.groupby('particle')['particle'].unique().index.values) for track in tracks)
    t_max = max(track['frame'].max() for track in tracks) + 1  # maximum frame number

    all_x = np.full((t_max, n_tracks), np.nan)
    all_y = np.full((t_max, n_tracks), np.nan)

    ntrack = 0
    for track in tracks:
        grouped = track.groupby('particle')
        particles = grouped['particle'].unique().index.values

        for i in range(0, len(particles)):
            df = grouped.get_group(particles[i])[['x', 'y', 'frame']].copy()
            frame_indices = np.array(df['frame'].astype(int))

            # fill the arrays with the values of the tracks
            all_x[frame_indices, ntrack] = df.loc[frame_indices]['x'].values
            all_y[frame_indices, ntrack] = df.loc[frame_indices]['y'].values
            ntrack = ntrack + 1
    return all_x, all_y

def plot_trajectories(all_x,all_y, plots_per_row=4, xlim=(-1000, 1000), ylim=(-1000, 1000), figsize=(20, 5)):
    """
    Plot the trajectories of particles with a maximum number of subplots per row.

    Parameters:
        all_x (numpy array): 2D numpy array where each column represents the x-coordinates of a trajectory.
        all_y (numpy array): 2D numpy array where each column represents the y-coordinates of a trajectory.
        plots_per_row (int): Maximum number of subplots per row.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        figsize (tuple): Size of the figure.

    Returns:
        None
    """
    all_x=np.copy(all_x)
    all_y=np.copy(all_y)
    num_plots = all_x.shape[1]
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(figsize[0], figsize[1] * num_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i in range(0,num_plots):
        x, y =all_x[:,i], all_y[:, i]
        x -=  x[np.where(~np.isnan(x))[0][0]]
        y -=  y[np.where(~np.isnan(y))[0][0]]
        axs[i].plot(x, y)
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].set_title(f'Trajectory {i}')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def timecolorplot_n(trajectory_x, trajectory_y, plots_per_row=4, xlim=(-800, 800),
                                ylim=(-800, 800), figsize=(20, 5)):
    """
    Plot time-color trajectories of particles with a maximum number of subplots per row.

    Parameters:
        trajectory_x (numpy array): 2D numpy array where each column represents the x-coordinates of a trajectory.
        trajectory_y (numpy array): 2D numpy array where each column represents the y-coordinates of a trajectory.
        time (numpy array): 1D numpy array representing the time steps.
        plots_per_row (int): Maximum number of subplots per row.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        figsize (tuple): Size of the figure.

    Returns:
        None
    """
    trajectory_x=np.copy(trajectory_x)
    trajectory_y=np.copy(trajectory_y)
    time = np.arange(np.shape(trajectory_x)[0])
    num_plots = trajectory_x.shape[1]
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(figsize[0], figsize[1] * num_rows))
    axs = np.array(axs).flatten()  # Flatten in case of multiple rows

    for i in range(num_plots):
        x, y = trajectory_x[:, i], trajectory_y[:, i]
        x -=  x[np.where(~np.isnan(x))[0][0]]
        y -= y[np.where(~np.isnan(y))[0][0]]

        xy = np.column_stack((x, y))
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        norm = plt.Normalize(time.min(), time.max())
        coll = LineCollection(segments, cmap=plt.cm.plasma, norm=norm)
        coll.set_array(time)
        coll.set_linewidth(1)

        ax = axs[i]
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'Trajectory {i}')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma), cax=cax)
        cax.set_title('Time')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def av_speed(all_x,all_y):
    #calculate the instantaneous velocity of the particles
    vel_mag=np.sqrt(np.diff(all_x,axis=0)**2+np.diff(all_y,axis=0)**2)
    #calculate the average velocity of the particles
    av_vel=np.nanmean(vel_mag,axis=0)
    return av_vel

# calculate the speeds to and plot distributions
def pdf_disp(x, y):
    dispx = np.diff(x, axis=0)
    dispy = np.diff(y, axis=0)
    # remove the NaN values
    dispx = np.copy(dispx[~np.isnan(dispx)])
    dispy = np.copy(dispy[~np.isnan(dispy)])
    disp = np.sqrt(dispx ** 2 + dispy ** 2)

    # Calculate the PDF for all rows in disp
    pdf = np.zeros((len(disp), 2))
    for i in range(len(disp)):
        pdf[i, 0], pdf[i, 1] = np.histogram(disp[i], bins=100)
    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the PDF
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, pdf, label='Probability Density')
    plt.xlabel('Speed')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Function of Speed')
    plt.legend()
    plt.show()


#%% To process all folders at once (not recommended)
for n in range(0,len(folders)):
    wt_files = files[folders[n]]
    wt_x, wt_y = get_tracks(wt_files)
    #function that calcualtes the average velocity of the particles
    wt_av_speed = av_speed(wt_x, wt_y) * 30 # in pixels/sec, videos are 30 fps

    #make a scatter plot of the average velocities comparing each group
    fig,ax=plt.subplots()
    ax.scatter(np.ones(len(wt_av_speed)), wt_av_speed, label='Control', color='blue')
    ax.plot(1, np.mean(wt_av_speed), color='blue', marker='x', markersize=20)
    ax.set_ylabel('Average Speed (pixel/s)')
    # ax.set_ylim(0, 2000)
    ax.legend()
    plt.show()

    timecolorplot_n(wt_x, wt_y)

    # remove the trajectory with an average speed smaller than 5 pixels/sec and save as a new file
    wt_x_filtered = wt_x[:,wt_av_speed > 5]
    wt_y_filtered = wt_y[:,wt_av_speed > 5]

    # Plot the filtered trajectories
    timecolorplot_n(wt_x_filtered, wt_y_filtered)

    # Reorganize the data
    # Extract all_x and all_y from data
    all_x = wt_x_filtered
    all_y = wt_y_filtered
    # Transpose before putting in track_array
    all_x = all_x.T
    all_y = all_y.T

    # Combine all_x and all_y into track_array such that the array is organized into (# of trajectories, # of frames, x&y coordinates)
    track_array = np.empty((all_x.shape[0], all_x.shape[1], 2))
    track_array[:, :, 0] = all_x
    track_array[:, :, 1] = all_y

    # Save the track_array as a .npy file
    np.save(os.path.join(folders[n], "filtered_trajectories.npy"), track_array)


#%% Process one file at a time to inspect and manually correct trajectories if needed (recommended)
folder_number = 0
wt_files = files[folders[folder_number]]  # Assuming you want to analyze the first folder
wt_x, wt_y = get_tracks(wt_files)

# function that calcualtes the average velocity of the particles
wt_av_speed = av_speed(wt_x, wt_y) * 30  # in pixels/sec, videos are 30 fps

# make a scatter plot of the average velocities comparing each group
fig, ax = plt.subplots()
ax.scatter(np.ones(len(wt_av_speed)), wt_av_speed, label='Control', color='blue')
ax.plot(1, np.mean(wt_av_speed), color='blue', marker='x', markersize=20)
ax.set_ylabel('Average Speed (pixel/s)')
# ax.set_ylim(0, 2000)
ax.legend()
plt.show()

timecolorplot_n(wt_x, wt_y)

# remove the trajectory with an average speed smaller than 5 pixels/sec and save as a new file
wt_x_filtered = wt_x[:, wt_av_speed > 5]
wt_y_filtered = wt_y[:, wt_av_speed > 5]
# for doublet and mouth videos that don't move much, use no speed filtering
# wt_x_filtered = wt_x
# wt_y_filtered = wt_y
# for doublet moving cells, which swim slower (change values if needed)
# wt_x_filtered = wt_x[:, wt_av_speed > 2.5]
# wt_y_filtered = wt_y[:, wt_av_speed > 2.5]

# Plot the filtered trajectories
timecolorplot_n(wt_x_filtered, wt_y_filtered)

# Reorganize the data
# Extract all_x and all_y from data
all_x = wt_x_filtered
all_y = wt_y_filtered
# Transpose before putting in track_array
all_x = all_x.T
all_y = all_y.T

# Combine all_x and all_y into track_array such that the array is organized into (# of trajectories, # of frames, x&y coordinates)
track_array = np.empty((all_x.shape[0], all_x.shape[1], 2))
track_array[:, :, 0] = all_x
track_array[:, :, 1] = all_y

# Save the track_array as a .npy file
np.save(os.path.join(folders[folder_number], "filtered_trajectories.npy"), track_array)






#%% If 2 trajectories are from the same cell, you can add them together manually. Use the rest of this code to do so.
# Look in the npy array when the trajectories are overlapping / changing from one to the other
start_timepoint = 8375  # frame where trajectories switch

# Replace trajectory 0 with trajectory 2 from timepoint 4265 onward
# track_array[0, start_timepoint:, :] = track_array[2, start_timepoint:, :]
track_array[0, start_timepoint:, :] = track_array[3, start_timepoint:, :]

# check if the trajectories are changed
all_x = track_array[:, :, 0]
all_y = track_array[:, :, 1]
all_x = all_x.T
all_y = all_y.T
wt_x_filtered = all_x
wt_y_filtered = all_y
# Plot the filtered trajectories
timecolorplot_n(wt_x_filtered, wt_y_filtered)

#%%
# Remove the unwanted trajectory after adding one to the other (otherwise double trajectory)
track_array = np.delete(track_array, 3, axis=0)

# check if the trajectories are changed
all_x = track_array[:, :, 0]
all_y = track_array[:, :, 1]
all_x = all_x.T
all_y = all_y.T
wt_x_filtered = all_x
wt_y_filtered = all_y
# Plot the filtered trajectories
timecolorplot_n(wt_x_filtered, wt_y_filtered)

#%%
# Save the new track_array as a .npy file
np.save(os.path.join(folders[folder_number], "filtered_trajectories.npy"), track_array)







