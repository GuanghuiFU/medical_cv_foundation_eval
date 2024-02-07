import os
import data_split
import binary_label

data_split_csv_path = '../brats-data_split.csv'
images_all_path = 'your/dataset/path/Task01_BrainTumour/imagesTr'
labels_all_path = 'your/dataset/path/Task01_BrainTumour/labelsTr'
images_tr_path = 'your/dataset/path/fm_experiment/imagesTr'
labels_tr_path = 'your/dataset/path/fm_experiment/labelsTr'
images_ts_path = 'your/dataset/path/fm_experiment/imagesTs'
labels_ts_path = 'your/dataset/path/fm_experiment/labelsTs'
os.makedirs(images_tr_path, exist_ok=True)
os.makedirs(labels_tr_path, exist_ok=True)
os.makedirs(images_ts_path, exist_ok=True)
os.makedirs(labels_ts_path, exist_ok=True)

# 1. Split dataset based on the csv file
data_split.main(images_all_path, labels_all_path, images_tr_path, labels_tr_path, images_ts_path, labels_ts_path, data_split_csv_path)

# 2. Binary label. The enhancing tumor is selected as the segmentation target
binary_label.main(labels_tr_path, labels_tr_path)
binary_label.main(images_ts_path, images_ts_path)