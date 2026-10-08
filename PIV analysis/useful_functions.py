import os
import numpy as np
import matplotlib.pyplot as plt
from fluidimage.data_objects.piv import MultipassPIVResults
import skimage.transform
import skimage.filters
from skimage import io
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt
import tifffile as tiff
from pystackreg import StackReg
from skimage.transform import resize

def load_fluidimage_fields(result_dir, pass_index=2):
    h5_files = sorted(
        os.path.join(result_dir, f)
        for f in os.listdir(result_dir)
        if f.endswith(".h5")
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in: {result_dir}")

    all_u = []
    all_v = []

    for h5_file in h5_files:
        result = MultipassPIVResults(h5_file)
        piv_attr = f"piv{pass_index}"
        if not hasattr(result, piv_attr):
            raise AttributeError(f"{h5_file} has no {piv_attr}")

        piv = getattr(result, piv_attr)
        xs, ys = piv.get_grid_pixel(pass_index)
        ny, nx = len(ys), len(xs)

        u = piv.deltaxs_final.reshape(ny, nx)
        v = piv.deltays_final.reshape(ny, nx)

        x_grid, y_grid = np.meshgrid(xs, ys)

        #correct the y axis so that the 0,0 is bottom left
        y_grid = y_grid.max() - y_grid
        v = -v

        all_u.append(u)
        all_v.append(v)

    return x_grid, y_grid, np.stack(all_u, axis=2), np.stack(all_v, axis=2)

def get_grid_points_in_mask_3d(mask_tyx, x_grid, y_grid, u, v, dilation_radius=20):
    nt = min(mask_tyx.shape[0], u.shape[2], v.shape[2])
    h, w = mask_tyx.shape[1:]

    # Preprocess mask: invert -> EDT dilation -> invert back
    mask_proc = mask_tyx.astype(bool).copy()

    for t in range(nt):
        dist_from_mask = distance_transform_edt(~mask_proc[t])
        mask_proc[t] = dist_from_mask <= dilation_radius
    mask_proc = ~mask_proc

    # grid (bottom-left coords) -> image indices (top-left coords)
    x_img = np.clip(np.rint(x_grid).astype(int), 0, w - 1)
    y_img = np.clip(np.rint((h - 1) - y_grid).astype(int), 0, h - 1)

    # (t, y, x) -> (t, ny, nx) -> (ny, nx, t)
    grid_keep = np.moveaxis(mask_proc[:nt, y_img, x_img].astype(bool), 0, 2)

    u_mask = np.full_like(u, np.nan, dtype=float)
    v_mask = np.full_like(v, np.nan, dtype=float)

    u_mask_t = u_mask[:, :, :nt]
    v_mask_t = v_mask[:, :, :nt]
    u_t = u[:, :, :nt]
    v_t = v[:, :, :nt]

    u_mask_t[grid_keep] = u_t[grid_keep]
    v_mask_t[grid_keep] = v_t[grid_keep]

    return u_mask, v_mask

def av_flow(window_sz, all_ugrid, all_vgrid):
    num_windows = all_ugrid.shape[2] - window_sz + 1
    u_avg = np.zeros((all_ugrid.shape[0], all_ugrid.shape[1], num_windows))
    v_avg = np.zeros((all_vgrid.shape[0], all_vgrid.shape[1], num_windows))

    for t in range(num_windows):
        u_avg[:, :, t] = np.mean(all_ugrid[:, :, t:t + window_sz], axis=2)
        v_avg[:, :, t] = np.mean(all_vgrid[:, :, t:t + window_sz], axis=2)

    return u_avg, v_avg


def colorplot(sc, norm):
    """Generates a colormap of the magnitude of a 2D vector field."""
    sz = np.multiply(norm.shape, sc) + sc
    sz = sz.astype(int)

    norm = np.nan_to_num(norm)  # replace NaNs before resizing
    disp_cmap = skimage.transform.resize_local_mean(
        norm, output_shape=sz, grid_mode=True, preserve_range=True
    )
    return disp_cmap

def save_flow_maps_as_gif(result_dir, u_avg, v_avg, x_grid, y_grid, units, vmax_um_s, quiver_step, quiver_scale):
    gif_path =result_dir#this has to include the complete filename, including the .gif extension

    x_unique = np.unique(x_grid)
    sc = int(np.median(np.diff(x_unique))) if len(x_unique) > 1 else 1

    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    xq = x_grid[::quiver_step, ::quiver_step]
    yq = y_grid[::quiver_step, ::quiver_step]

    u0 = u_avg[:, :, 0]
    v0 = v_avg[:, :, 0]
    speed0 = np.hypot(u0, v0) * units
    av_speed0 = colorplot(sc, speed0)

    fig, ax = plt.subplots()
    im = ax.imshow(av_speed0, cmap="viridis", vmin=0, vmax=vmax_um_s, origin="upper", extent=extent)
    fig.colorbar(im, ax=ax, label="Velocity (um/s)")
    q = ax.quiver(xq, yq, u0[::quiver_step, ::quiver_step], v0[::quiver_step, ::quiver_step], color="w", scale=quiver_scale)
    ax.axis("off")

    with imageio.get_writer(gif_path, mode="I", duration=0.1) as writer:
        for t in range(u_avg.shape[2]):
            u_t = u_avg[:, :, t]
            v_t = v_avg[:, :, t]
            speed_t = np.hypot(u_t, v_t) * units
            av_speed = colorplot(sc, speed_t)

            im.set_data(av_speed)
            q.set_UVC(u_t[::quiver_step, ::quiver_step], v_t[::quiver_step, ::quiver_step])

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    plt.close(fig)
    print(f"Saved flow gif: {gif_path}")

def register_full_stack_from_cropped(stack_path,transform_name,crop_top,crop_left):

    full_stack = tiff.imread(stack_path)
    cropped_stack = tiff.imread(stack_path.replace(".tif", "_cropped.tif"))

    sr = StackReg(transform_name)
    tmats = sr.register_stack(cropped_stack, reference="previous", axis=0)

    c = np.array([[1.0, 0.0, -crop_left],
                  [0.0, 1.0, -crop_top],
                  [0.0, 0.0, 1.0]])
    c_inv = np.array([[1.0, 0.0, crop_left],
                      [0.0, 1.0, crop_top],
                      [0.0, 0.0, 1.0]])
    tmats_full = np.array([c_inv @ t @ c for t in tmats])

    registered_full = sr.transform_stack(full_stack, tmats=tmats_full, axis=0)
    registered_cropped = sr.transform_stack(cropped_stack, tmats=tmats,axis=0)

    registered_full = np.clip(np.rint(registered_full), 0, 255).astype(np.uint8)
    registered_cropped = np.clip(np.rint(registered_cropped), 0, 255).astype(np.uint8)

    return registered_full,registered_cropped,tmats_full

def register_full_stack_from_scaled(stack_path,transform_name,scale=0.25):
    full_stack = tiff.imread(stack_path)

    sr = StackReg(transform_name)

    t, h, w = full_stack.shape
    small_h = int(round(h * scale))
    small_w = int(round(w * scale))

    small_stack = resize( full_stack,(t, small_h, small_w),order=0,preserve_range=True,anti_aliasing=False).astype(full_stack.dtype)
    tmats = sr.register_stack(small_stack, reference="previous", axis=0)

    sy = small_h / h
    sx = small_w / w

    c = np.array([[sx, 0.0, 0.0],
                  [0.0, sy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    c_inv = np.array([[1.0/sx, 0.0, 0.0],
                      [0.0, 1.0/sy, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    tmats_full = np.array([c_inv @ tmat @ c for tmat in tmats])

    registered_full = sr.transform_stack(full_stack, tmats=tmats_full, axis=0)
    registered_scaled = sr.transform_stack(small_stack, tmats=tmats, axis=0)

    registered_full = (registered_full > 0).astype(np.uint8)
    registered_scaled = (registered_scaled > 0).astype(np.uint8)

    return registered_full,registered_scaled,tmats_full