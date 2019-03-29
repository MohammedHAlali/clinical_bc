# Clinical Feature Extraction from Whole-Slide Images based on Pathology Reports

This repository contains supplementary materials for our paper 'Clinical feature extraction from whole-slide images based on pathology reports'

How to reproduce:
1. Run [0_read_resize.py]() code, which will 
    - Read images (113 png files) in 'images/' folder. Each image corresponds to a breast cancer case.
    - Resize them to (2048, 4069)
    - Create three more copies, by flipping, for each image
    - Saves all four images associated with each bresat cancer case.
    - At the end, this code will output .npy files in 'npy/' directory
