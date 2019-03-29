# Clinical Feature Extraction from Whole-Slide Images based on Pathology Reports

This repository contains supplementary materials for our paper 'Clinical feature extraction from whole-slide images based on pathology reports'

How to reproduce:
First: run '0_read_resize.py' code, which will 
  A- read images (113 png files) in 'images/' folder. Each image corresponds to a breast cancer case.
  B- resize them to (2048, 4069)
  C- create three more copies, by flipping, for each image
  D- saves all four images associated with each bresat cancer case.
  E- At the end, this code will output .npy files in 'npy/' directory
