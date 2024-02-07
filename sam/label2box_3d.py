import os
import nibabel as nib
import numpy as np
import pandas as pd


def calculate_3d_coordinates_and_size(folder_path, excel_save_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz'):
            file_path = os.path.join(folder_path, filename)

            img = nib.load(file_path)
            data = img.get_fdata()

            indices = np.argwhere(data == 1)
            if indices.size > 0:
                min_coords = np.min(indices, axis=0)
                max_coords = np.max(indices, axis=0)
                size = max_coords - min_coords + 1
                results.append({
                    'Filename': filename,
                    'Position': f"{min_coords[0]},{min_coords[1]},{min_coords[2]}",
                    'Size': f"{size[0]},{size[1]},{size[2]}"
                })

    df = pd.DataFrame(results)
    output_file = os.path.join(folder_path, excel_save_path)
    df.to_csv(output_file, index=False)

    return output_file

def main(label_path, excel_save_path):
    output_file = calculate_3d_coordinates_and_size(label_path, excel_save_path)
    print(f"Results saved to csv: {output_file}")
