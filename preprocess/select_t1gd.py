import os
import glob
import nibabel as nib


def load_and_save_t1gd_nifti(folder_path, output_folder):
    # Define the modality of interest
    modality_index = 2  # T1gd is the third modality (0-indexed)
    # Process each file
    for file_path in glob.glob(os.path.join(folder_path, '*.nii.gz')):
        img = nib.load(file_path)
        data = img.get_fdata()

        # Check if the image has the expected shape
        if data.shape != (240, 240, 155, 4):
            print(f"Skipping {os.path.basename(file_path)}, unexpected shape: {data.shape}")
            continue

        # Extract and save only the T1gd modality data
        modality_data = data[:, :, :, modality_index]
        modality_img = nib.Nifti1Image(modality_data, img.affine)
        modality_name = os.path.basename(file_path).replace('.nii.gz','_0000.nii.gz')
        modality_file_path = os.path.join(output_folder, modality_name)
        nib.save(modality_img, modality_file_path)

    print("Processing complete.")


def main(folder_path, output_folder):
    load_and_save_t1gd_nifti(folder_path, output_folder)
