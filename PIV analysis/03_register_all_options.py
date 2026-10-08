from pystackreg import StackReg
import pandas as pd
import os
import numpy as np
import tifffile as tiff
from useful_functions import register_full_stack_from_cropped, register_full_stack_from_scaled

#registers based on a cropped section of the image. cropped_coordinates.csv is a csv file with the coordinates of the cropped section of the image.

base_path = "/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/"
masks_path = os.path.join(base_path, "masks")
crop_coords = pd.read_csv(os.path.join(masks_path, "cropped_coordinates.csv"))
crop_coords["short_name"] = crop_coords["short_name"].str.replace(".tif", "_masks.tif", regex=False)

files = [f for f in os.listdir(masks_path) if f.endswith("_cropped.tif")]
files = [f.replace("_cropped.tif", ".tif") for f in files]


tmats_path = os.path.join(base_path, "tmats")
os.makedirs(tmats_path, exist_ok=True)
sanity_path = os.path.join(tmats_path, "sanity_checks")
os.makedirs(sanity_path, exist_ok=True)

stackreg_transform = StackReg.RIGID_BODY

print(f"Found {len(files)} files to process in {masks_path}.")

print(crop_coords)

for file_name in files:
    print(file_name)
    if file_name not in crop_coords["short_name"].values:
        print(f"Skipping {file_name} because it is not in cropped_coordinates.csv")
        continue

    stack_path = os.path.join(masks_path, file_name)
    crop_top  = crop_coords.loc[crop_coords["short_name"] == file_name, "y"].values[0]
    crop_left = crop_coords.loc[crop_coords["short_name"] == file_name, "x"].values[0]

    print(crop_top)
    print(crop_left)

    registered_full, _, tmats_full = register_full_stack_from_cropped(stack_path, stackreg_transform, crop_top, crop_left)
    np.save(os.path.join(tmats_path, file_name[:-4] + "_transform.npy"), tmats_full)
    tiff.imwrite(os.path.join(sanity_path, file_name[:-4] + "_aligned.tif"), registered_full, imagej=True, metadata={"axes": "TYX"})

