import os
import numpy as np
from skimage.filters import difference_of_gaussians
from skimage import exposure
import pims
import tifffile as tiff

@pims.pipeline
def dog_filter(frame):
    filtered = difference_of_gaussians(frame, 2, 12)
    filtered = exposure.rescale_intensity(filtered, out_range=(0, 65535)).astype(frame.dtype)
    return filtered

path = "/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/"
#list files that end in aligned.tif
files=[f for f in os.listdir(path + "raw_data/") if f.endswith(".tif")]

#make a list that combines the path and files
files = [os.path.join(path + "raw_data/", f) for f in files]
#make a new directory for the filtered images
output_dir = os.path.join(path, "filtered")
os.makedirs(output_dir, exist_ok=True)

print(f"Found {len(files)} files to process in {path}.")

for file in files:
    input_file = file
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".tif", "_filt.tif"))

    #read input image
    frames = pims.open(input_file)

    #apply dog_filter and save stack
    dog = dog_filter(frames)
    #explicitly convert to uint16 to avoid issues with tifffile
    dog = np.asarray([frame for frame in dog_filter(frames)], dtype=np.uint16)
    tiff.imwrite(output_file, dog, photometric='minisblack')


