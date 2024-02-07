import os
import nibabel as nib

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, file_name)

            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()

            img_data[img_data == 1] = 0
            img_data[img_data == 2] = 1
            img_data[img_data == 3] = 1

            new_img = nib.Nifti1Image(img_data, nii_img.affine, nii_img.header)

            output_file_path = os.path.join(output_folder, file_name)

            nib.save(new_img, output_file_path)

