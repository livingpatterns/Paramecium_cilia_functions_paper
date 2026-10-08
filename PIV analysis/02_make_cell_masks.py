import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

path = "/Users/kourkoul/SCRIPTS/PIV_Daphne_analysis/"

#find all files in folder that end in .filt
file_names = [f for f in os.listdir(path+'filtered/') if os.path.isfile(os.path.join(path+'filtered/', f)) and f.endswith('_filt.tif')]
#make a new directory for the masks
masks_path=os.path.join(path,"masks")
os.makedirs(masks_path, exist_ok=True)

for file in file_names:
    im_stack=tiff.imread(os.path.join(path+'filtered/', file))
    mask_stack=np.zeros(im_stack.shape,dtype=np.uint8)

    for idx, im in enumerate(im_stack):
        predictor, segmenter = get_predictor_and_segmenter(model_type="vit_b_lm")#model_type="vit_b_lm"
        cell_mask = automatic_instance_segmentation(predictor=predictor,segmenter=segmenter,input_path=im)
        mask_stack[idx] = np.asarray(cell_mask, dtype=np.uint8)

    tiff.imwrite(os.path.join(masks_path, os.path.splitext(file)[0] + "_masks.tif"), mask_stack,imagej=True, metadata={"axes": "TYX"})