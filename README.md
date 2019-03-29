# Clinical Feature Extraction from Whole-Slide Images based on Pathology Reports

This repository contains supplementary materials for our paper 'Clinical feature extraction from whole-slide images based on pathology reports'

How to reproduce:
1. run '0_read_resize.py' code, which will 
  - read images (113 png files) in 'images/' folder. Each image corresponds to a breast cancer case.
  - resize them to (2048, 4069)
  - create three more copies, by flipping, for each image
  - saves all four images associated with each bresat cancer case.
  - At the end, this code will output .npy files in 'npy/' directory
