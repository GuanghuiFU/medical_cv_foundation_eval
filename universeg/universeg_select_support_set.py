import numpy as np
import pandas as pd
import os
from shutil import copyfile
import matplotlib.pyplot as plt


def select_slice(subject_names, label_dir, support_set_dir, selection='largest', min_area=10):
    support_images_path = os.path.join(f'{support_set_dir}_{selection}', 'images')
    support_labels_path = os.path.join(f'{support_set_dir}_{selection}', 'labels')

    # Ensure the support directories exist
    os.makedirs(support_images_path, exist_ok=True)
    os.makedirs(support_labels_path, exist_ok=True)

    for subject in subject_names:
        subject_label_dir = os.path.join(label_dir, subject)
        file_selection = None
        area_to_file_map = []

        try:
            # Iterate through all slice files in the subject's label directory
            for file_name in os.listdir(subject_label_dir):
                if file_name.endswith('.png'):
                    file_path = os.path.join(subject_label_dir, file_name)
                    label_image = plt.imread(file_path)
                    current_sum_or_area = np.sum(label_image) if selection != 'middle' else np.sum(label_image > 0)

                    if selection == 'middle' and current_sum_or_area > 0:
                        area_to_file_map.append((current_sum_or_area, file_path))
                    else:
                        if (selection == 'largest' and current_sum_or_area > (file_selection[0] if file_selection else 0)) or \
                           (selection == 'smallest' and 0 < current_sum_or_area < (file_selection[0] if file_selection else np.inf) and current_sum_or_area > min_area):
                            file_selection = (current_sum_or_area, file_path)

            # Process selection based on criteria
            if selection == 'middle' and area_to_file_map:
                area_to_file_map.sort(key=lambda x: x[0])
                middle_index = len(area_to_file_map) // 2
                if len(area_to_file_map) % 2 == 0 and len(area_to_file_map) > 1:
                    middle_index -= 1
                file_selection = area_to_file_map[middle_index]

            if file_selection:
                selected_file_path = file_selection[1]
                slice_num = int(os.path.basename(selected_file_path).split('_')[1])
                label_dest_file_path = f'{support_labels_path}/{subject}_{slice_num}.png'
                image_dest_file_path = f'{support_images_path}/{subject}_{slice_num}.png'

                # Copy label
                copyfile(selected_file_path, label_dest_file_path)
                # Copy corresponding MRI image
                image_file_path = selected_file_path.replace('labels', 'images')
                copyfile(image_file_path, image_dest_file_path)

                print(f"Copied {selected_file_path} to {label_dest_file_path}")
            else:
                print(f"No suitable label files found for subject: {subject} using criteria '{selection}'")
        except Exception as e:
            print(f"An error occurred while processing labels for subject {subject} using criteria '{selection}': {e}")

def main(data_split_csv_path, label_dir, support_set_base_dir, selection):
    data_split_csv_path = '../brats-data_split.csv'
    df = pd.read_csv(data_split_csv_path)
    train_subjects_df = df[df['Train/Test'] == 'Train']
    subject_names = train_subjects_df['File'].tolist()
    '''
    Note that the label_dir here is the 2d slice of your label. The structure of this folder should be like this:
    
    slice_128_128
    ├── imagesTr_slice
    │   ├── BRATS_076
    │   │   ├── slice_0_image.png
    │   │   ├── slice_1_image.png
    │   │   ├── slice_..._image.png
    │   │   └── slice_154_image.png
    │   └── BRATS_077
    │       ├── slice_0_image.png
    │       ├── slice_1_image.png
    │       ├── slice_..._image.png
    │       └── slice_154_image.png
    ├── imagesTs_slice
    │   ├── BRATS_001
    │   │   ├── slice_0_image.png
    │   │   ├── slice_1_image.png
    │   │   ├── slice_..._image.png
    │   │   └── slice_154_image.png
    │   └── BRATS_077
    │       ├── slice_0_image.png
    │       ├── slice_1_image.png
    │       ├── slice_..._image.png
    │       └── slice_154_image.png
    ├── labelsTr_slice
    │   ├── BRATS_076
    │   │   ├── slice_0_image.png
    │   │   ├── slice_1_image.png
    │   │   ├── slice_..._image.png
    │   │   └── slice_154_image.png
    │   └── BRATS_077
    │       ├── slice_0_image.png
    │       ├── slice_1_image.png
    │       ├── slice_..._image.png
    │       └── slice_154_image.png
    └── labelsTs_slice
        ├── BRATS_076
        │   ├── slice_0_image.png
        │   ├── slice_1_image.png
        │   ├── slice_..._image.png
        │   └── slice_154_image.png
        └── BRATS_077
            ├── slice_0_image.png
            ├── slice_1_image.png
            ├── slice_..._image.png
            └── slice_154_image.png
    '''
    
    label_dir = 'your/2d/level/dataset/path/slice_128_128/labelsTr_slice'
    support_set_base_dir = 'your/support/set/path'
    select_slice(subject_names, label_dir, support_set_base_dir, selection=selection, min_area=10)
    # select_slice(subject_names, label_dir, support_set_base_dir, selection='largest', min_area=10)
    # select_slice(subject_names, label_dir, support_set_base_dir, selection='smallest', min_area=10)
    # select_slice(subject_names, label_dir, support_set_base_dir, selection='middle', min_area=10)
