#!/bin/bash

python inpainting_SDXL.py \
    --base_dir output_with_All_parameters_learnable22_0.2 \
    --base_path /path/to/base_path \
    --image_directory /path/to/image_directory \
    --mask_directory /path/to/mask_directory \
    --embs_path /path/to/embs_path