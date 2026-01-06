import os
import re
import nd2
import pims
import numpy as np
import tifffile as tiff
from microfilm import colorify
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor

def max_proj_nd2(hyperstack, path, name):
    """
    Create a max projection of a multichannel image and save it as a TIFF file.

    Parameters:
    hyperstack (numpy.ndarray): The multichannel image array.
    path (str): The path to the directory where the TIFF file will be saved.
    name (str): The name of the TIFF file.

    Returns:
    numpy.ndarray: The max projection image array.
    """
    # Determine the dimensions of the hyperstack
    dims = np.shape(hyperstack)  # Z, channels, y, x

    def process_stack(stack):
        dna = stack[:, 0, :]  # All Z slices of channel 1
        centrin = stack[:, 2, :]  # All Z slices of channel 2
        polye = stack[:, 1, :]  # All Z slices of channel 3

        # Create max projections of the channels
        max_centrin = np.max(centrin, axis=0)
        max_polye = np.max(polye, axis=0)
        max_dna = np.max(dna, axis=0)

        return np.stack((max_dna.astype(np.float32), max_centrin.astype(np.float32), max_polye.astype(np.float32)), axis=0)

    # Check the number of dimensions and process accordingly
    if len(dims) > 4:
        with ThreadPoolExecutor() as executor:
            three_channel_images = list(executor.map(process_stack, [hyperstack[i] for i in range(dims[0])]))

        os.makedirs(path + 'max_proj', exist_ok=True)
        for i, image in enumerate(three_channel_images):
            tiff.imwrite(
                path + 'max_proj/' + name[:-4] + f"_max_proj_{i}.tif",
                image,
                imagej=True,
                metadata={
                    'axes': 'CYX',
                    'channel_colors': [
                        {'color': 'Red'},
                        {'color': 'Green'},
                        {'color': 'Blue'}
                    ]
                }
            )
    elif len(dims) == 4:
        three_channel_image = process_stack(hyperstack)
        os.makedirs(path + 'max_proj', exist_ok=True)
        tiff.imwrite(
            path + 'max_proj/' + name[:-4] + "_max_proj.tif",
            three_channel_image,
            imagej=True,
            metadata={
                'axes': 'CYX',
                'channel_colors': [
                    {'color': 'Red'},
                    {'color': 'Green'},
                    {'color': 'Blue'}
                ]
            }
        )


# # WHEN 2 CHANNELS ONLY, CENTRIN + POLYE
def max_proj_nd2_2ch(hyperstack, path, filename):
    """
    Create a max projection of a 2-channel image and save it as a TIFF file.

    Parameters:
    hyperstack (numpy.ndarray): The multichannel image array with shape (Z, C, Y, X).
    path (str): The path to the directory where the TIFF file will be saved.
    name (str): The name of the TIFF file.

    Returns:
    numpy.ndarray: The max projection image array.
    """
    dims = np.shape(hyperstack)  # (Z, C, Y, X)

    def process_stack(stack):
        dna = stack[:, 0, :]      # Channel 0 → DNA
        centrin = stack[:, 1, :]  # Channel 1 → Centrin

        max_dna = np.max(dna, axis=0)
        max_centrin = np.max(centrin, axis=0)

        # Output 2-channel stacked image: [DNA (Red), Centrin (Green)]
        return np.stack((
            max_dna.astype(np.float32),
            max_centrin.astype(np.float32)
        ), axis=0)

    os.makedirs(os.path.join(path, 'max_proj'), exist_ok=True)

    if len(dims) > 4:
        with ThreadPoolExecutor() as executor:
            projections = list(executor.map(process_stack, [hyperstack[i] for i in range(dims[0])]))

        for i, image in enumerate(projections):
            tiff.imwrite(
                os.path.join(path, 'max_proj', name[:-4] + f"_max_proj_{i}.tif"),
                image,
                imagej=True,
                metadata={
                    'axes': 'CYX',
                    'channel_colors': [
                        {'color': 'Red'},
                        {'color': 'Green'}
                    ]
                }
            )
    elif len(dims) == 4:
        projection = process_stack(hyperstack)
        tiff.imwrite(
            os.path.join(path, 'max_proj', name[:-4] + "_max_proj.tif"),
            projection,
            imagej=True,
            metadata={
                'axes': 'CYX',
                'channel_colors': [
                    {'color': 'Red'},
                    {'color': 'Green'}
                ]
            }
        )



def max_proj_nd2_dynamic(hyperstack, path, name):
    """
    Create a max projection for 2- or 3-channel ND2 files and save it as a TIFF.

    Parameters:
    hyperstack (numpy.ndarray): Image array of shape (Z, C, Y, X)
    path (str): Save directory
    name (str): Original filename

    Returns:
    numpy.ndarray: Max-projected image array
    """
    import os
    import numpy as np
    import tifffile as tiff
    from concurrent.futures import ThreadPoolExecutor

    dims = np.shape(hyperstack)  # Expecting (Z, C, Y, X)
    num_channels = dims[1]

    def process_stack(stack):
        dna = np.max(stack[:, 0, :], axis=0)
        centrin = np.max(stack[:, 1, :], axis=0)

        if num_channels == 3:
            polye = np.max(stack[:, 2, :], axis=0)
            return np.stack((
                dna.astype(np.float32),
                centrin.astype(np.float32),
                polye.astype(np.float32)
            ), axis=0)
        elif num_channels == 2:
            return np.stack((
                dna.astype(np.float32),
                centrin.astype(np.float32)
            ), axis=0)
        else:
            raise ValueError(f"Unsupported channel count: {num_channels}")

    os.makedirs(os.path.join(path, 'max_proj'), exist_ok=True)

    if len(dims) > 4:
        with ThreadPoolExecutor() as executor:
            projections = list(executor.map(process_stack, [hyperstack[i] for i in range(dims[0])]))

        for i, image in enumerate(projections):
            tiff.imwrite(
                os.path.join(path, 'max_proj', name[:-4] + f"_max_proj_{i}.tif"),
                image,
                imagej=True,
                metadata={
                    'axes': 'CYX',
                    'channel_colors': [
                        {'color': 'Red'},
                        {'color': 'Green'}
                    ] + ([{'color': 'Blue'}] if num_channels == 3 else [])
                }
            )
    elif len(dims) == 4:
        projection = process_stack(hyperstack)
        tiff.imwrite(
            os.path.join(path, 'max_proj', name[:-4] + "_max_proj.tif"),
            projection,
            imagej=True,
            metadata={
                'axes': 'CYX',
                'channel_colors': [
                    {'color': 'Red'},
                    {'color': 'Green'}
                ] + ([{'color': 'Blue'}] if num_channels == 3 else [])
            }
        )


# Import and classify nd2 files into a dictionary, divided by sample and extracting the calibration data from the metadata file
def import_and_classify_nd2_files(path):
    """
    Import all the nd2 files in a folder, classify them by sample based on info extracted from the filename,
    and store them in a dictionary as numpy arrays.

    :param path: Path to the folder containing the nd2 files
    :param ext: Extension of the files to be imported (e.g., '.nd2').
    :param keyword: Keyword to filter the files (e.g., '3016').
    :return: Dictionary with the classified nd2 files as numpy arrays.
    """
    samplegroup = {}
    centrin = {}
    polye = {}
    dna = {}

    # Define a pims pipeline to ensure images are converted to numpy arrays
    @pims.pipeline
    def as_gray(centrin):
        return centrin[:, :, 0]  # Assuming the image is grayscale

    # Loop through all the files in the folder
    for filename in os.listdir(path):
        # Check if the file is an nd2 file
        if filename.endswith('.nd2'):
            # Extract the sample info from the filename
            match = re.search(r'\d{6}_(.+?)_cell\d+',
                              filename)  # Match the pattern "date_<info(cell_type)>_cell<number>"
            if match:
                sample_info = match.group(1)  # Extract the info (e.g., "AED")
            else:
                sample_info = "Unknown"  # Default category if no match is found

            with nd2.ND2File(path + filename) as nd2_file:
                metadata = nd2_file.frame_metadata(0)
                calibration = metadata.channels[0].volume.axesCalibration[0]
                # sz_value = metadata.channels[0].volume.voxelCount
            hyperstack = nd2.imread(path + filename)
            max_proj_nd2(hyperstack, path, filename)

            # Add the nd2 file to the dictionary under the appropriate sample category
            if sample_info not in samplegroup:
                samplegroup[sample_info] = []
            samplegroup[sample_info].append((filename, hyperstack, calibration))

            hyperstack = nd2.imread(path + filename)
            # Extract the channels from the hyperstack
            centrin_image = np.max(hyperstack[:, 1, :], axis=0)  # All Z slices of channel 2
            polye_image = np.max(hyperstack[:, 2, :], axis=0)  # All Z slices of channel 3
            dna_image = np.max(hyperstack[:, 0, :], axis=0)  # All Z slices of channel 1

            if centrin_image.ndim == 3:
                centrin_image = as_gray(centrin_image)
            if polye_image.ndim == 3:
                polye_image = as_gray(polye_image)
            if dna_image.ndim == 3:
                dna_image = as_gray(dna_image)

            if sample_info not in centrin:
                centrin[sample_info] = []
            centrin[sample_info].append((filename, centrin_image, calibration))  # projecting the centrin channel
            if sample_info not in polye:
                polye[sample_info] = []
            polye[sample_info].append((filename, polye_image, calibration))  # projecting the polye channel
            if sample_info not in dna:
                dna[sample_info] = []
            dna[sample_info].append((filename, dna_image, calibration))  # projecting the dna channel

    print(f"Imported {len(samplegroup)} samples from {path}.")
    print(f"Samples found: {list(samplegroup.keys())}")

    return samplegroup, centrin, polye, dna




# Function to process 2-channel ND2 files
def import_and_classify_nd2_files_2ch(path):
    """
    Import all the 2-channel ND2 files in a folder, classify them by sample based on info extracted from the filename,
    and store them in a dictionary as numpy arrays.

    :param path: Path to the folder containing the nd2 files.
    :return: Dictionaries with the classified nd2 files and individual channels.
    """
    samplegroup = {}
    centrin = {}
    dna = {}

    @pims.pipeline
    def as_gray(image):
        return image[:, :, 0]  # Convert to grayscale if 3D

    for filename in os.listdir(path):
        if filename.endswith('.nd2'):
            match = re.search(r'\d{6}_(.+?)_cell\d+', filename)
            sample_info = match.group(1) if match else "Unknown"

            with nd2.ND2File(os.path.join(path, filename)) as nd2_file:
                metadata = nd2_file.frame_metadata(0)
                calibration = metadata.channels[0].volume.axesCalibration[0]

            hyperstack = nd2.imread(os.path.join(path, filename))
            # max_proj_nd2_dynamic(hyperstack, path, filename)
            max_proj_nd2_2ch(hyperstack, path, filename)

            # Store the full file
            if sample_info not in samplegroup:
                samplegroup[sample_info] = []
            samplegroup[sample_info].append((filename, hyperstack, calibration))

            # Extract channels: assuming 0 = DNA, 1 = Centrin
            dna_image = np.max(hyperstack[:, 0, :], axis=0)
            centrin_image = np.max(hyperstack[:, 1, :], axis=0)

            if dna_image.ndim == 3:
                dna_image = as_gray(dna_image)
            if centrin_image.ndim == 3:
                centrin_image = as_gray(centrin_image)

            if sample_info not in dna:
                dna[sample_info] = []
            dna[sample_info].append((filename, dna_image, calibration))

            if sample_info not in centrin:
                centrin[sample_info] = []
            centrin[sample_info].append((filename, centrin_image, calibration))

    print(f"Imported {len(samplegroup)} samples from {path}.")
    print(f"Samples found: {list(samplegroup.keys())}")

    return samplegroup, centrin, dna



# Function to display the multichannel images
def display_multichannel_samplegroup(samplegroup):
    """
    Display the multichannel images in a subplot.
    Each row represents a different sample, and each column shows the images for that sample.
    """
    for sample, files in samplegroup.items():
        for filename, hyperstack, calibration in files:
            dna = hyperstack[:, 0, :]  # all z slices of ch1
            centrin = hyperstack[:, 1, :]  # all z slices of ch2
            polye = hyperstack[:, 2, :]  # all z slices of chs3

            max_centrin = np.max(centrin, axis=0)
            max_polye = np.max(polye, axis=0)
            max_dna = np.max(dna, axis=0)

            # Show the tiff images as a check, using colorify for choosing the color and improving the contrast
            showcentrin, cmap, min_max = colorify.colorify_by_name(max_centrin, cmap_name='magenta', flip_map=False)
            showpolye, cmap, min_max = colorify.colorify_by_hex(max_polye, cmap_hex='#F5EC05')
            showdna, cmap, min_max = colorify.colorify_by_hex(max_dna, cmap_hex='#00ffff')

            # Show the images with colorify
            plt.figure(1, figsize=(28, 8))
            plt.subplot(1, 4, 1)
            plt.imshow(showcentrin)
            plt.title('Centrin Channel')
            plt.axis('off')

            # Add the scale
            scale_length_px = 50 / calibration
            scale_x_start = 50  # Posizione iniziale della scala (in pixel)
            scale_y_start = showcentrin.shape[0] - 50  # Posizione verticale (vicino al bordo inferiore)
            scale_bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                                          linewidth=0, edgecolor=None, facecolor='white')
            plt.gca().add_patch(scale_bar)

            # Add the text next to the scale
            plt.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, f'{50} μm',
                     color='white', fontsize=10, ha='center', va='bottom')

            plt.subplot(1, 4, 2)
            plt.imshow(showpolye)
            plt.title('PolyE Channel')

            # Add the scale
            scale_length_px = 50 / calibration
            scale_x_start = 50  # Posizione iniziale della scala (in pixel)
            scale_y_start = showpolye.shape[0] - 50  # Posizione verticale (vicino al bordo inferiore)
            scale_bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                                          linewidth=0, edgecolor=None, facecolor='white')
            plt.gca().add_patch(scale_bar)

            # Add the text next to the scale
            plt.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, f'{50} μm',
                     color='white', fontsize=10, ha='center', va='bottom')
            plt.axis('off')
            plt.subplot(1, 4, 3)
            plt.imshow(showdna)
            plt.title('DNA Channel')
            plt.subplot(1, 4, 4)
            # Add the scale
            scale_length_px = 50 / calibration
            scale_x_start = 50  # Posizione iniziale della scala (in pixel)
            scale_y_start = showdna.shape[0] - 50  # Posizione verticale (vicino al bordo inferiore)
            scale_bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                                          linewidth=0, edgecolor=None, facecolor='white')
            plt.gca().add_patch(scale_bar)
            plt.axis('off')
            # Add the text next to the scale
            plt.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, f'{50} μm',
                     color='white', fontsize=10, ha='center', va='bottom')

            combined = colorify.combine_image([showcentrin, showpolye, showdna])
            plt.imshow(combined)
            # Add the scale
            scale_length_px = 50 / calibration
            scale_x_start = 50  # Posizione iniziale della scala (in pixel)
            scale_y_start = showcentrin.shape[0] - 50  # Posizione verticale (vicino al bordo inferiore)
            scale_bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                                          linewidth=0, edgecolor=None, facecolor='white')
            plt.gca().add_patch(scale_bar)

            # Add the text next to the scale
            plt.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, f'{50} μm',
                     color='white', fontsize=10, ha='center', va='bottom')
            plt.axis('off')
            plt.title('All Channels')
            # Title of the whole figure
            plt.suptitle(f'Max Intensity Projections of {filename}', fontsize=20)
            plt.tight_layout()
            plt.show()



def display_multichannel_samplegroup_2ch(samplegroup):
    """
    Display 2-channel ND2 images (Centrin & DNA) in a subplot.
    Each image is shown with color mapping and scale bar.
    """
    for sample, files in samplegroup.items():
        for filename, hyperstack, calibration in files:
            # Extract and max-project channels
            dna = np.max(hyperstack[:, 0, :], axis=0)
            centrin = np.max(hyperstack[:, 1, :], axis=0)

            # Apply colorify to each channel
            showcentrin, _, _ = colorify.colorify_by_name(centrin, cmap_name='magenta', flip_map=False)
            showdna, _, _ = colorify.colorify_by_hex(dna, cmap_hex='#00ffff')

            # Set up figure
            plt.figure(figsize=(24, 6))

            # Centrin
            plt.subplot(1, 3, 1)
            plt.imshow(showcentrin)
            plt.title('Centrin Channel')
            plt.axis('off')
            _add_scale_bar(showcentrin.shape, calibration)

            # DNA
            plt.subplot(1, 3, 2)
            plt.imshow(showdna)
            plt.title('DNA Channel')
            plt.axis('off')
            _add_scale_bar(showdna.shape, calibration)

            # Combined
            plt.subplot(1, 3, 3)
            combined = colorify.combine_image([showcentrin, showdna])
            plt.imshow(combined)
            plt.title('Combined Channels')
            plt.axis('off')
            _add_scale_bar(combined.shape, calibration)

            # Add main title
            plt.suptitle(f'Max Intensity Projections of {filename}', fontsize=20)
            plt.tight_layout()
            plt.show()



# Helper function to draw a 50 μm scale bar
def _add_scale_bar(shape, calibration):
    scale_length_px = 50 / calibration
    scale_x_start = 50
    scale_y_start = shape[0] - 50
    bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                            linewidth=0, facecolor='white')
    plt.gca().add_patch(bar)
    plt.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, '50 μm',
             color='white', fontsize=10, ha='center', va='bottom')




def display_samplegroup_with_scale(samplegroup, scale_length_um=50):
    import matplotlib.pyplot as plt

    # Determine the number of rows and columns for the subplots
    num_rows = len(samplegroup)
    num_cols = max(len(files) for files in samplegroup.values())

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Ensure `axes` is always a 2D array for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = np.expand_dims(axes, axis=0 if num_rows == 1 else 1)

    # Iterate over the samplegroup and display the images
    for row_idx, (sample, files) in enumerate(samplegroup.items()):
        for col_idx, (filename, centrin, calibration) in enumerate(files):
            ax = axes[row_idx][col_idx]
            ax.imshow(centrin, cmap='gray')  # Display the image
            ax.set_title(f"{sample}\n{filename}", fontsize=10)
            ax.axis('off')

            # Add a scale bar
            scale_length_px = scale_length_um / calibration
            ax.plot(
                [10, 10 + scale_length_px], [centrin.shape[0] - 10, centrin.shape[0] - 10],
                color='white', linewidth=2
            )
            ax.text(
                10, centrin.shape[0] - 20, f"{scale_length_um} μm",
                color='white', fontsize=8, verticalalignment='bottom'
            )

    # Hide unused subplots
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if row_idx >= len(samplegroup) or col_idx >= len(samplegroup[list(samplegroup.keys())[row_idx]]):
                axes[row_idx][col_idx].axis('off')

    plt.tight_layout()
    plt.show()



# Function to apply simple pre-processing function to each image of the samplegroup
def apply_function_to_samplegroup(samplegroup, func, **kwargs):

    """
    Apply a given function to all images in a samplegroup dictionary and create a new dictionary.

    :param samplegroup: Dictionary containing sample data in the format:
                        {sample_name: [(filename, image, calibration), ...]}
    :param func: Function to apply to each image.
    :param kwargs: Additional keyword arguments to pass to the function.
    :return: New dictionary with the same structure as samplegroup, but with the function applied to the images.
    """
    new_samplegroup = {}

    for sample, files in samplegroup.items():
        for filename, image, calibration in files:
            # Apply the function to the image
            processed_image = func(image, **kwargs)

            # Add the processed image to the new dictionary
            if sample not in new_samplegroup:
                new_samplegroup[sample] = []
            new_samplegroup[sample].append((filename, processed_image, calibration))

    return new_samplegroup



def display_samplegroup(samplegroup):
    """
    Display the dictionary of classified tif files in a subplot.
    Each row represents a different sample, and each column shows the images for that sample.
    """
    # Determine the number of rows and columns for the subplots
    num_rows = len(samplegroup)
    num_cols = max(len(files) for files in samplegroup.values())

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

    # Ensure `axes` is always a 2D array for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = np.expand_dims(axes, axis=0 if num_rows == 1 else 1)

    # Iterate over the samplegroup and display the images
    for row_idx, (sample, files) in enumerate(samplegroup.items()):
        for col_idx, (filename, centrin, calibration) in enumerate(files):
            ax = axes[row_idx][col_idx]
            ax.imshow(centrin, cmap='gray')  # Display the image
            ax.set_title(f"{sample}\n{filename}", fontsize=10)
            ax.axis('off')

    # Hide unused subplots
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if row_idx >= len(samplegroup) or col_idx >= len(samplegroup[list(samplegroup.keys())[row_idx]]):
                axes[row_idx][col_idx].axis('off')

    plt.tight_layout()
    plt.show()



# Function to display the dictionary with a legend under each image
def display_samplegroup_with_legend(samplegroup):
    """
    Display the dictionary of classified labeled images in a subplot with a legend under each image.
    Each row represents a different sample, and each column shows the labeled images for that sample.
    """
    num_samples = len(samplegroup)  # Number of sample categories
    max_images_per_row = max(len(files) for files in samplegroup.values())  # Max number of images in a row

    # Create the figure
    fig, axes = plt.subplots(num_samples, max_images_per_row, figsize=(5 * max_images_per_row, 6 * num_samples))

    # Loop through each sample and its images
    for row_idx, (sample, files) in enumerate(samplegroup.items()):
        for col_idx, (filename, labeled_image, num_features, calibration) in enumerate(files):
            ax = axes[row_idx][col_idx] if num_samples > 1 else axes[col_idx]
            ax.imshow(labeled_image, cmap='bone')  # Display the labeled image
            ax.axis('off')

            # Add the sample name and filename as the title
            ax.set_title(f"{sample}\n{filename}", fontsize=10, loc='center')

            # Add a legend under the image
            regions = measure.regionprops(labeled_image)
            legend_labels = [f"Cell {region.label}" for region in regions]
            legend_text = "\n".join(legend_labels)
            ax.text(0.5, -0.1, legend_text, transform=ax.transAxes, fontsize=8, ha='center', va='top', color='black')

        # Hide empty axes if there are fewer images than the max number of columns
        for col_idx in range(len(files), max_images_per_row):
            if num_samples > 1:
                axes[row_idx][col_idx].axis('off')
            else:
                axes[col_idx].axis('off')

    plt.tight_layout()
    plt.show()



# Display the filtered labeled images
def display_samplegroup_with_labels(samplegroup, scale_length_um=50):
    """
    Display the dictionary of classified labeled images in a subplot with labels.
    Each row represents a different sample, and each column shows the labeled images for that sample.
    """
    num_samples = len(samplegroup)  # Number of sample categories
    max_images_per_row = max(len(files) for files in samplegroup.values())  # Max number of images in a row

    # Create the figure
    fig, axes = plt.subplots(num_samples, max_images_per_row, figsize=(5 * max_images_per_row, 5 * num_samples))

    # Loop through each sample and its images
    for row_idx, (sample, files) in enumerate(samplegroup.items()):
        for col_idx, (filename, labeled_image, num_features, calibration) in enumerate(files):
            ax = axes[row_idx][col_idx] if num_samples > 1 else axes[col_idx]
            ax.imshow(labeled_image, cmap='bone')  # Display the labeled image
            if col_idx == 0:  # Show the sample name only in the first column
                ax.set_title(f"{sample}", fontsize=14, fontweight='bold', loc='left')
            ax.set_xlabel(f"{filename}", fontsize=12)  # Show the filename
            ax.axis('off')

            # Add labels (numbers) on top of each region
            regions = measure.regionprops(labeled_image)
            for region in regions:
                y, x = region.centroid  # Coordinates of the centroid
                ax.text(x, y, str(region.label), color='red', fontsize=12, fontweight='bold', ha='center', va='center')

        # Hide empty axes if there are fewer images than the max number of columns
        for col_idx in range(len(files), max_images_per_row):
            if num_samples > 1:
                axes[row_idx][col_idx].axis('off')
            else:
                axes[col_idx].axis('off')

    plt.tight_layout()
    plt.show()



def calculate_mean_properties(masked_filtered_images):
    """
    Calculate the mean area and mean major axis length for each sample in the masked_filtered_images dictionary.

    :param masked_filtered_images: Dictionary containing labeled and filtered images for each sample.
    :return: Dictionary with mean area and mean major axis length for each sample.
    """
    mean_properties = {}

    for sample, files in masked_filtered_images.items():
        total_area = 0
        total_major_axis_length = 0
        total_regions = 0

        for filename, labeled_image, num_features, calibration in files:
            # Calculate region properties
            regions = measure.regionprops(labeled_image)

            # Accumulate area and major axis length
            for region in regions:
                total_area += region.area * (calibration ** 2)
                total_major_axis_length += region.major_axis_length * calibration
                total_regions += 1

        # Calculate mean properties for the sample
        if total_regions > 0:
            mean_area = total_area / total_regions
            mean_major_axis_length = total_major_axis_length / total_regions
        else:
            mean_area = 0
            mean_major_axis_length = 0

        mean_properties[sample] = {
            "Mean Area": mean_area,
            "Mean Major Axis Length": mean_major_axis_length
        }

    return mean_properties



# Display the images with mean properties
def display_samplegroup_with_mean_properties(samplegroup, mean_properties, scale_length_um=50):
    """
    Display the dictionary of classified labeled images in a subplot with mean properties for each sample.
    Each row represents a different sample, and each column shows the labeled images for that sample.

    :param samplegroup: Dictionary containing labeled and filtered images for each sample.
    :param mean_properties: Dictionary with mean area and mean major axis length for each sample.
    :param scale_length_um: Length of the scale bar in micrometers.
    """
    num_samples = len(samplegroup)  # Number of sample categories
    max_images_per_row = max(len(files) for files in samplegroup.values())  # Max number of images in a row

    # Create the figure
    fig, axes = plt.subplots(num_samples, max_images_per_row, figsize=(5 * max_images_per_row, 6 * num_samples))

    # Loop through each sample and its images
    for row_idx, (sample, files) in enumerate(samplegroup.items()):
        for col_idx, (filename, labeled_image, num_features, calibration) in enumerate(files):
            ax = axes[row_idx][col_idx] if num_samples > 1 else axes[col_idx]
            ax.imshow(labeled_image, cmap='bone')  # Display the labeled image
            ax.axis('off')

            # Add the sample name and filename as the title
            if col_idx == 0:  # Show the sample name only in the first column
                mean_area = mean_properties[sample]["Mean Area"]
                mean_major_axis_length = mean_properties[sample]["Mean Major Axis Length"]
                ax.set_title(
                    f"{sample}\nMean Area: {mean_area:.2f} μm²\nMean Major Axis: {mean_major_axis_length:.2f} μm",
                    fontsize=10,
                    loc='left'
                )

            # Add a scale bar
            scale_length_px = scale_length_um / calibration
            scale_x_start = 50  # Starting position of the scale (in pixels)
            scale_y_start = labeled_image.shape[0] - 50  # Vertical position (near the bottom edge)
            scale_bar = patches.Rectangle((scale_x_start, scale_y_start), scale_length_px, 5,
                                           linewidth=0, edgecolor=None, facecolor='white')
            ax.add_patch(scale_bar)

            # Add the text next to the scale bar
            ax.text(scale_x_start + scale_length_px / 2, scale_y_start - 10, f'{scale_length_um} μm',
                    color='white', fontsize=10, ha='center', va='bottom')

        # Hide empty axes if there are fewer images than the max number of columns
        for col_idx in range(len(files), max_images_per_row):
            if num_samples > 1:
                axes[row_idx][col_idx].axis('off')
            else:
                axes[col_idx].axis('off')

    plt.tight_layout()
    plt.show()

