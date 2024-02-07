import os
import preprocess.data_split as data_split
import preprocess.binary_label as binary_label
import preprocess.select_t1gd as select_t1gd

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

# Pre-step: download the dataset from medical segmentation decathlon website: https://drive.google.com/file/d/1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU/view?usp=drive_link

# 1. Split dataset based on the csv file
data_split.main(images_all_path, labels_all_path, images_tr_path, labels_tr_path, images_ts_path, labels_ts_path, data_split_csv_path)

# 2. Get T1-GD MRI from MSD-BraTs dataset
select_t1gd.main(images_tr_path, images_tr_path)
select_t1gd.main(images_ts_path, images_ts_path)

# 2. Binary label. The enhancing tumor is selected as the segmentation target
binary_label.main(labels_tr_path, labels_tr_path)
binary_label.main(images_ts_path, images_ts_path)