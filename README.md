# Clinical Feature Extraction from Whole-Slide Images based on Pathology Reports

This repository contains supplementary materials for our paper 'Clinical feature extraction from whole-slide images based on pathology reports'

*How to reproduce:
1.: run '0_read_resize.py' code, which will 
1.1- read images (113 png files) in 'images/' folder. Each image corresponds to a breast cancer case.
1.2- resize them to (2048, 4069)
1.3- create three more copies, by flipping, for each image
1.4- saves all four images associated with each bresat cancer case.
1.5- At the end, this code will output .npy files in 'npy/' directory
