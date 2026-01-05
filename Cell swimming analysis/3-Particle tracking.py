"""
# Particle Tracking for cell swimming video analysis
This script links detected particles across video frames to reconstruct trajectories of swimming cells.
It processes particle position data from the previous particle locating code, applies TrackPy to associate particles between frames, and outputs the resulting trajectories and visualizations.

Input:
- path: Directory containing analysis folders with detected particle positions (`all_raw_particle_positions.pkl`)
- search_range: Maximum distance (in pixels) a particle can move between frames
- memory: Maximum number of frames a particle can be lost and still be tracked
- threshold: Minimum number of frames a particle must be tracked to be considered valid
- To process one video at a time, set video_files and analysis_folders to lists of length 1 with the desired video and analysis folder paths (recommended to check each video individually). Specify these paths directly in the code in video_files and analysis_folders variables.

Output:
- Pickle (`.pkl`) files with tracked particle trajectories, saved in each analysis folder
- PNG plots of all tracks, saved in the `plots` subdirectory of each analysis folder
- A text file with tracking parameters used (`parameters_tracking_tp.txt`)
"""

import trackpy as tp
import os
import pandas as pd


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


# Specify the path with your .pkl files where particle detections are stored
path = 'W:/Users/Daphne/Imaging_Daphne/25-11-18_RPi_ptetwt_swimming_deciliated/bgd_subs_4000/'

#list of folders that end in _analysis
analysis_folders = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_analysis')]
video_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mp4')])

# #to test things one video at a time
video_files=path+'processed_181125_ptetwt_swimming_singlets_2.mp4'
video_files=[video_files]
analysis_folders=path+'processed_181125_ptetwt_swimming_singlets_2_analysis'
analysis_folders=[analysis_folders]


for n_video in range(0,len(video_files)):
    analysis_path=analysis_folders[n_video]
    video_name = os.path.basename(analysis_path.split('_', 1)[1].rsplit('_', 1)[0])  # name of the video
    print(video_name)
    #Tracking with trackpy
    #import .pkl file with particle positions
    track_me=pd.read_pickle(analysis_path+'/all_raw_particle_positions.pkl')

    #Track the particles using trackpy
    search_range= 50 #maximum distance in pixels that a particle can move between frames
    memory= 10 #maximum number of frames a particle can be lost and still be tracked
    threshold= 1800 #minimum number of frames a particle has to be tracked to be considered good

    tp.quiet()  # turn off progress reports for now
    tracks = tp.link(track_me, search_range=search_range, memory=memory)
    pruned_tracks = tp.filter_stubs(tracks, threshold=threshold)#remove tracks that are shorter than 800 frames
    tp.plot_traj(pruned_tracks)
    if not pruned_tracks.empty:
        ax=tp.plot_traj(pruned_tracks)
        fig=ax.get_figure()
        # Set equal scaling for x and y axes
        ax.set_aspect('equal', adjustable='box')
        #add title with the name of the video
        ax.set_title(video_name)
        #save plotted tracks in analysis folder
        fig.savefig(analysis_path+'/plots/all_tracks_tp.png')
        #save the tracks if dataframe is not empty

        pruned_tracks.to_pickle(analysis_path+'/all_tracks_tp.pkl')

    #save the parameters used for tracking in a text file
    with open(analysis_path+'/parameters_tracking_tp.txt', 'w') as f:
        f.write('search_range='+str(search_range)+'\n')
        f.write('memory='+str(memory)+'\n')
        f.write('threshold='+str(threshold)+'\n')
        f.close()
