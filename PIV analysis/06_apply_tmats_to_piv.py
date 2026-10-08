import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tifffile as tiff

path = "/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/"

names = sorted(name[:-4] for name in os.listdir(path+'filtered/') if name.endswith('.tif'))


file_names  = [f"{n}.tif" for n in names]
mask_names  = [os.path.join(path,"masks",f"{n[:-5]}_filt_masks.tif") for n in names]
tmats_names = [os.path.join(path,"tmats",f"{n[:-5]}_filt_transform.npy") for n in names]
piv_names = [os.path.join(path, "piv_as_npy", f"{n}_piv.npz") for n in names]

for piv_path, tmats_path, mask_path, name in zip(piv_names, tmats_names, mask_names, names):
    output_path = os.path.join(path, "piv_as_npy", f"{name}_aligned_piv.npz")

    piv_data = np.load(piv_path)
    tmats = np.load(tmats_path)
    mask_tyx = tiff.imread(mask_path)
    H = mask_tyx.shape[1]

    x = piv_data["x_grid"]
    y = piv_data["y_grid"]
    u = piv_data["u_masked"]
    v = piv_data["v_masked"]

    x_orig = x.copy()
    y_orig = y.copy()

    u_out = np.full_like(u, np.nan, dtype=float)
    v_out = np.full_like(v, np.nan, dtype=float)

    ny, nx = x.shape
    x_img = x
    y_img = H - 1 - y

    if np.any(np.diff(y_img[:, 0]) < 0):
        y_axis = y_img[::-1, 0]
        flip = True
    else:
        y_axis = y_img[:, 0]
        flip = False

    x_axis = x_img[0]
    pts = np.stack([x_img.ravel(), y_img.ravel(), np.ones(x.size)])
    n_frames = min(tmats.shape[0], u.shape[2])

    for k in range(n_frames):
        T = tmats[k]
        src = T @ pts

        x_src = src[0].reshape(ny, nx)
        y_src = src[1].reshape(ny, nx)

        u_img = u[:, :, k]
        v_img = -v[:, :, k]

        if flip:
            u_img = u_img[::-1, :]
            v_img = v_img[::-1, :]

        interp_u = RegularGridInterpolator((y_axis, x_axis), u_img, bounds_error=False, fill_value=np.nan)
        interp_v = RegularGridInterpolator((y_axis, x_axis), v_img, bounds_error=False, fill_value=np.nan)

        sample_points = np.column_stack([y_src.ravel(), x_src.ravel()])
        u_src = interp_u(sample_points).reshape(ny, nx)
        v_src = interp_v(sample_points).reshape(ny, nx)

        R = T[:2, :2]
        uv = np.stack([u_src.ravel(), v_src.ravel()])
        uv_rot = R @ uv

        u_out[:, :, k] = uv_rot[0].reshape(ny, nx)
        v_out[:, :, k] = -uv_rot[1].reshape(ny, nx)
    np.savez(output_path,x_grid=x_orig,y_grid=y_orig,all_u=u,all_v=v,u_aligned=u_out,v_aligned=v_out)
    print(f"Saved {name}")

