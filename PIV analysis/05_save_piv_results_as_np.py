import os
import numpy as np
import tifffile as tiff
from useful_functions import get_grid_points_in_mask_3d,load_fluidimage_fields

path = "/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/"

names = sorted(name[:-4] for name in os.listdir(path + 'filtered/') if name.endswith('.tif'))

file_names = [f"{n}.tif" for n in names]
mask_names = [os.path.join(path, "masks", f"{n[:-5]}_filt_masks.tif") for n in names]
tmats_names = [os.path.join(path, "tmats", f"{n[:-5]}_transform.npy") for n in names]
piv_names = [os.path.join(path, "piv_results", f"{n}_piv_results") for n in names]


pass_index = 2 #this is since it starts counting at 0 so third pass is index 2
output_dir = os.path.join(path, "piv_as_npy")
os.makedirs(output_dir, exist_ok=True)

for piv_path, mask_path, name in zip(piv_names, mask_names, names):
    x_grid, y_grid, u, v = load_fluidimage_fields(piv_path,pass_index=pass_index)
    mask_tyx = tiff.imread(mask_path)
    u_masked, v_masked = get_grid_points_in_mask_3d(mask_tyx, x_grid, y_grid, u, v)

    np.savez(os.path.join(output_dir, f"{name}_piv.npz"), x_grid=x_grid, y_grid=y_grid, all_u=u, all_v=v,
             u_masked=u_masked, v_masked=v_masked)

    print(f"Saved PIV results for {name}")

