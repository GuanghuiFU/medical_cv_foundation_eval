import pandas as pd
import os
import shutil


# Read CSV file

# Function to move files
def move_files(row, images_source_path, labels_source_path, images_dest_path, labels_dest_path):
    file_base = row['File']
    # Assuming the image and label files follow a specific naming convention
    # Update these lines if the file extensions or naming conventions differ
    image_file_name = file_base + ".nii.gz"
    label_file_name = file_base + ".nii.gz"

    # Construct source and destination paths
    image_source = os.path.join(images_source_path, image_file_name)
    label_source = os.path.join(labels_source_path, label_file_name)
    image_dest = os.path.join(images_dest_path, image_file_name)
    label_dest = os.path.join(labels_dest_path, label_file_name)

    # Move the files
    if os.path.exists(image_source) and os.path.exists(label_source):
        shutil.copy(image_source, image_dest)
        shutil.copy(label_source, label_dest)
    else:
        print(f"Files for {file_base} not found in source directories.")


def main(images_all_path, labels_all_path, images_tr_path, labels_tr_path, images_ts_path, labels_ts_path,
         data_split_csv_path):
    data_split = pd.read_csv(data_split_csv_path)
    # Iterate over each row in the dataframe
    for index, row in data_split.iterrows():
        if row['Train/Test'] == 'Train':
            move_files(row, images_all_path, labels_all_path, images_tr_path, labels_tr_path)
        elif row['Train/Test'] == 'Test':
            move_files(row, images_all_path, labels_all_path, images_ts_path, labels_ts_path)
        else:
            print(f"Invalid category for {row['File']}")

    print("Data moving completed.")
