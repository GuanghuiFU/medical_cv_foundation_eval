import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

def calculate_rectangle_coordinates_and_size_per_file(folder_path, excel_folder_path):
    if not os.path.exists(excel_folder_path):
        os.makedirs(excel_folder_path)

    for file_path in glob.glob(os.path.join(folder_path, '*.nii.gz')):
        results = []
        img = nib.load(file_path)
        data = img.get_fdata()
        for slice_index in range(data.shape[2]):
            slice_data = data[:,:,slice_index]
            indices = np.argwhere(slice_data > 0)

            if indices.size > 0:
                min_coords = np.min(indices, axis=0)
                max_coords = np.max(indices, axis=0)
                size = max_coords - min_coords + 1
                results.append({
                    'SliceNo': slice_index,
                    'Position': f"{min_coords[0]},{min_coords[1]}",
                    'Size': f"{size[0]},{size[1]}"
                })

        base_filename = os.path.basename(file_path).replace('.nii.gz', '_slice_prompt.csv')
        excel_file = os.path.join(excel_folder_path, base_filename)
        df = pd.DataFrame(results)
        df.to_csv(excel_file, index=False)

    print(f"Excel files saved in {excel_folder_path}")

def main(label_path, excel_save_folder_path):
    calculate_rectangle_coordinates_and_size_per_file(label_path, excel_save_folder_path)
    print(f"Results saved to folder: {excel_save_folder_path}")

