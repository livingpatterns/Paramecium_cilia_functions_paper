import os
import numpy as np
import tifffile as tiff
os.environ["OMP_NUM_THREADS"] = "1"
from fluidimage.piv import Topology
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
from fluidimage.data_objects.piv import MultipassPIVResults

def export_stack_to_frames(stack_path, dst_dir, frame_step=1):
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    os.makedirs(dst_dir, exist_ok=True)

    # remove old exported frames
    for f in os.listdir(dst_dir):
        if f.endswith(".tif"):
            os.remove(os.path.join(dst_dir, f))

    stack = tiff.imread(stack_path)
    for out_i, src_i in enumerate(range(0, len(stack), frame_step)):
        frame = stack[src_i]
        tiff.imwrite(os.path.join(dst_dir, f"frame_{out_i:06d}.tif"), frame.astype(np.uint16))

#for informaiton on what these parameters are see: https://fluidimage.readthedocs.io/en/latest/_generated/fluidimage.topologies.piv.html

#run PIV for all files that end in _filt.tif
path = "C:/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/filtered"
# files_to_process = [f for f in os.listdir(os.path.dirname(path)) if f.endswith("_filt.tif")]
files_to_process = [f for f in os.listdir(path) if f.endswith("_filt.tif")]
#files_to_process = ['/Users/guillermina/Desktop/piv_test/tack_cells/filtered/cell20_flow_filt.tif']
print(f"Found {len(files_to_process)} files to process in {path}.")
for file in files_to_process:
    input_path = os.path.join(path,file)
    output_dir = os.path.join(os.path.dirname(input_path),os.path.splitext(os.path.basename(input_path))[0]+'_piv_results')
    glob_pattern = "*.tif"
    str_subset = "i:i+2"

    window_size = 128
    overlap = 0.5
    passes = 3
    zoom = 2
    use_tps = "all" #interpolation method for multipass: "last" or "all" or False
    subdom_size : int = window_size/2#size of the interpolation window
    smoothing_coef : float = 10#5 is often reasonable. Can typically be between 0 to 40.
    threshold_tps =5# #Allowed difference of displacement (in pixels) between smoothed and input

    correl_min = 0.2#0.5
    threshold_diff_neighbour = 8#4
    displacement_max_piv0 = 15#15
    displacement_max = 10

    sequential = False#this has to do with code parallelization
    # -------------------------------------------

    im_seq_dir = os.path.join(output_dir, "im_seq")
    export_stack_to_frames(input_path, im_seq_dir, frame_step=1 )

    params = Topology.create_default_params()
    params.series.path = os.path.join(im_seq_dir, glob_pattern)
    params.series.str_subset = str_subset

    params.piv0.shape_crop_im0 = window_size
    params.piv0.grid.overlap = overlap
    params.piv0.displacement_max = displacement_max_piv0

    params.fix.correl_min = correl_min
    params.fix.threshold_diff_neighbour = threshold_diff_neighbour
    params.fix.displacement_max = displacement_max


    params.multipass.number = passes
    if passes > 1:
        params.multipass.coeff_zoom = zoom
    params.multipass.use_tps = "all" if use_tps else False
    params.multipass.subdom_size = subdom_size
    params.multipass.smoothing_coef = smoothing_coef
    params.multipass.threshold_tps = threshold_tps


    params.saving.how = "recompute"
    params.saving.path = output_dir
    params.saving.postfix = "fluidimage_piv"

    topology = Topology(params)
    topology.compute(sequential=sequential)

    print(f"Done. Results written in: {topology.path_dir_result}")

#this saves vector field visualizations, useful when debugging the piv parameters
    # result_dir = str(topology.path_dir_result)
    # post_dir = os.path.join(result_dir, "post")
    # os.makedirs(post_dir, exist_ok=True)

    # h5_files = sorted(
    #     os.path.join(result_dir, f)
    #     for f in os.listdir(result_dir)
    #     if f.endswith(".h5")
    # )

    # for h5_file in h5_files:
    #     base = os.path.splitext(os.path.basename(h5_file))[0]
    #     result = MultipassPIVResults(h5_file)

    #     # 1) Built-in display: vectors over image (includes non-interpolated/spurious vectors view)
    #     disp = result.display(show_interp=False, show_error=True,hist=False)
    #     fig_overlay = disp.figure if hasattr(disp, "figure") else disp.fig
    #     fig_overlay.savefig(os.path.join(post_dir, f"{base}_vectors_on_image.png"), dpi=200, bbox_inches="tight")
    #     plt.close(fig_overlay)
    # print(f"Saved vector overlays in: {post_dir}")

