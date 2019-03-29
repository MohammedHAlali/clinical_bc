# Clinical Feature Extraction from Whole-Slide Images based on Pathology Reports

This repository contains supplementary materials for our paper 'Clinical feature extraction from whole-slide images based on pathology reports'

How to reproduce:
1. Run [0_read_resize.py](0_read_resize.py) code, which will do the following:
  - Read images (113 png files) in 'images/' folder. Each image corresponds to a breast cancer case.
  - Resize them to (2048, 4069)
  - Create three more copies, by flipping, for each image
  - Saves all four images associated with each bresat cancer case.
  - At the end, this code should output .npy files in 'npy/' directory, which will be created by the code.
2. Make sure the two files for features are available: [extracted_features_csv.csv](extracted_features_csv.csv) and [nationwidechildrens.org_clinical_patient_brca.txt](nationwidechildrens.org_clinical_patient_brca.txt) in the same working directory. Then, run [1_read_clinical_features.py](1_read_clinical_features.py) code which will do the following:
  - For each of the 100 breast cancer case in [extracted_features_csv.csv](extracted_features_csv.csv), it will gather few more related features from [nationwidechildrens.org_clinical_patient_brca.txt](nationwidechildrens.org_clinical_patient_brca.txt) about the same cae.
  - The feature's value is converted into binary format.
  - The final binary formatted features are stored in a csv file.
  - At the end, this code should output the csv file called: 'clean_features.csv'
