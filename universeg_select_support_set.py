import numpy as np
import os
from shutil import copyfile
import matplotlib.pyplot as plt


def load_subject_names(file_path):
    try:
        subject_names = np.load(file_path, allow_pickle=True)
        return subject_names.tolist()
    except Exception as e:
        print(f"An error occurred while loading subject names: {e}")
        return []


def select_largest_slices(subject_names, label_dir, support_set_dir):
    support_images_path = os.path.join(support_set_dir, 'images')
    support_labels_path = os.path.join(support_set_dir, 'labels')

    for subject in subject_names:
        subject_label_dir = os.path.join(label_dir, subject)
        max_sum = 0
        largest_label_file = None
        try:
            # Iterate through all slice files in the subject's label directory
            for file_name in os.listdir(subject_label_dir):
                if file_name.endswith('.png'):
                    file_path = os.path.join(subject_label_dir, file_name)
                    label_image = plt.imread(file_path)
                    current_sum = np.sum(label_image)
                    if current_sum > max_sum:
                        max_sum = current_sum
                        largest_label_file = file_path
            # If a largest label file was found, copy it to the support set
            if largest_label_file:
                slice_num = int(os.path.basename(largest_label_file).split('_')[1])
                dest_file_path = f'{support_labels_path}/{subject}_{slice_num}.png'
                copyfile(largest_label_file, dest_file_path)
                largest_image_file = largest_label_file.replace('label','MRI')
                dest_images_file_path = f'{support_images_path}/{subject}_{slice_num}.png'
                copyfile(largest_image_file, dest_images_file_path)

                print(f"Copied {largest_label_file} to {dest_file_path}")
            else:
                print(f"No label files found for subject: {subject}")
        except Exception as e:
            print(f"An error occurred while processing labels for subject {subject}: {e}")


def select_smallest_slices(subject_names, label_dir, support_set, min_area=10):
    support_images_path = os.path.join(support_set, 'images')
    support_labels_path = os.path.join(support_set, 'labels')
    if not os.path.exists(support_images_path):
        os.makedirs(support_images_path)
    if not os.path.exists(support_labels_path):
        os.makedirs(support_labels_path)
    for subject in subject_names:
        subject_label_dir = os.path.join(label_dir, subject)
        min_sum = np.inf  # Set the initial minimum sum to infinity
        smallest_label_file = None
        try:
            # Iterate through all slice files in the subject's label directory
            for file_name in os.listdir(subject_label_dir):
                if file_name.endswith('.png'):
                    file_path = os.path.join(subject_label_dir, file_name)
                    label_image = plt.imread(file_path)
                    current_sum = np.sum(label_image)
                    # Check if the current_sum is smaller than min_sum, greater than zero and above min_area threshold
                    if 0 < current_sum < min_sum and current_sum > min_area:
                        min_sum = current_sum
                        smallest_label_file = file_path
            # If a smallest label file was found, copy it to the support set
            if smallest_label_file:
                slice_num = int(os.path.basename(smallest_label_file).split('_')[1])
                dest_file_path = f'{support_labels_path}/{subject}_{slice_num}.png'
                copyfile(smallest_label_file, dest_file_path)
                smallest_image_file = smallest_label_file.replace('label', 'MRI')
                dest_images_file_path = f'{support_images_path}/{subject}_{slice_num}.png'
                copyfile(smallest_image_file, dest_images_file_path)

                print(f"Copied {smallest_label_file} to {dest_file_path}")
            else:
                print(f"No suitable label files found for subject: {subject}")
        except Exception as e:
            print(f"An error occurred while processing labels for subject {subject}: {e}")


def select_middle_area_slice(subject_names, label_dir, support_set_dir):
    support_images_path = os.path.join(support_set_dir, 'images')
    support_labels_path = os.path.join(support_set_dir, 'labels')
    if not os.path.exists(support_images_path):
        os.makedirs(support_images_path)
    if not os.path.exists(support_labels_path):
        os.makedirs(support_labels_path)
    for subject in subject_names:
        subject_label_dir = os.path.join(label_dir, subject)
        area_to_file_map = []
        try:
            # Iterate through all slice files in the subject's label directory
            for file_name in os.listdir(subject_label_dir):
                if file_name.endswith('.png'):
                    file_path = os.path.join(subject_label_dir, file_name)
                    label_image = plt.imread(file_path)
                    current_area = np.sum(label_image > 0)  # Area is the count of non-zero pixels
                    if current_area > 0:
                        area_to_file_map.append((current_area, file_path))

            if area_to_file_map:
                # Sort by area
                area_to_file_map.sort(key=lambda x: x[0])
                # Find the middle value index
                middle_index = len(area_to_file_map) // 2
                # If there are an even number of slices, take the second of the middle two
                if len(area_to_file_map) % 2 == 0:
                    middle_index = middle_index if len(area_to_file_map) > 1 else 0

                # Select the file with the middle area
                middle_area_file = area_to_file_map[middle_index][1]
                slice_num = int(os.path.basename(middle_area_file).split('_')[1])
                dest_file_path = f'{support_labels_path}/{subject}_{slice_num}.png'
                copyfile(middle_area_file, dest_file_path)
                middle_image_file = middle_area_file.replace('label', 'MRI')
                dest_images_file_path = f'{support_images_path}/{subject}_{slice_num}.png'
                copyfile(middle_image_file, dest_images_file_path)

                print(f"Copied slice of middle area {middle_area_file} to {dest_file_path}")
            else:
                print(f"No non-zero label files found for subject: {subject}")
        except Exception as e:
            print(f"An error occurred while processing labels for subject {subject}: {e}")


train_npy_path = 'your/train/set/numpy/list/train.npy'
label_dir = 'your/label/path'
support_set_dir = 'your/path/to/save/support/set'
subject_names = load_subject_names(train_npy_path)
select_largest_slices(subject_names, label_dir, support_set_dir)
select_smallest_slices(subject_names, label_dir, support_set_dir)
select_middle_area_slice(subject_names, label_dir, support_set_dir)
