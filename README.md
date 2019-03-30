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
3. Run [2_combine_img_label.py](2_combine_img_label.py) code, which will do the following:
  - Read all images (.npy) and associate them with their labels, i.e. list of 49 features.
  - At the end, this code should output three data files:
    - all 452 images in one file, i.e. 'data_images.npz'
    - all (corresponding) labels in one file, i.e. 'data_labels.npz'
    - A third file, i.e. 'data_labels.csv' that are used to count the number of feature count in all dataset.
4. Run [3_create_fold_data.py](3_create_fold_data.py) code, which will do the following:
  - Read the two data files, split them into 5 folds of training and testing data.
  - At the end, 4x5=20 data files will be created.
5. Run [4_tf_cnn_model.py](4_tf_cnn_model.py) code as follows:
  - python 4_tf_cnn_model.py exp 1 lr 0.1 epo 1 batch 8 fold 1
  - This will run CNN model for training and testing.
  - At the end, results are shown in csv files.
  - For all 5 folds, take the average result of all 5 numbers.
