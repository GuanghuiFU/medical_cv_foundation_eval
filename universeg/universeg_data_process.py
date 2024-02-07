import os
from PIL import Image
from scipy.ndimage import zoom
import nibabel as nib
import numpy as np


def nifti_to_png_slices(image_path, output_folder, resize_shape=(128, 128)):
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()
    zoom_factor = (resize_shape[0] / img_data.shape[0], resize_shape[1] / img_data.shape[1], 1)
    img_resized = zoom(img_data, zoom_factor, order=3)
    for idx in range(img_resized.shape[2]):
        img_slice = img_resized[:, :, idx]
        img_pil = Image.fromarray((img_slice / img_slice.max() * 255).astype('uint8'), 'L')
        img_pil.save(os.path.join(output_folder, f'slice_{idx}_image.png'))


def nifti_to_png_slices_binary(image_path, output_folder, resize_shape=(128, 128)):
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()
    zoom_factor = (resize_shape[0] / img_data.shape[0], resize_shape[1] / img_data.shape[1], 1)
    img_resized = zoom(img_data, zoom_factor, order=0)  # Using nearest-neighbor since it's label data
    img_resized = np.round(img_resized).astype('uint8')
    for idx in range(img_resized.shape[2]):
        img_slice = img_resized[:, :, idx]
        img_pil = Image.fromarray((img_slice * 255).astype('uint8'), 'L')  # multiply by 255 to convert 1s to 255
        img_pil.save(os.path.join(output_folder, f'slice_{idx}_image.png'))


def mri_3d_2d_slice(image_dir, save_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            image_path = os.path.join(image_dir, filename)
            output_folder = os.path.join(save_dir, filename.split('.')[0])
            os.makedirs(output_folder, exist_ok=True)
            try:
                nifti_to_png_slices(image_path, output_folder)
            except:
                print('[ERROR]:', filename)


def label_3d_2d_slice(label_dir, save_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            label_path = os.path.join(label_dir, filename)
            output_folder = os.path.join(save_dir, filename.split('.')[0])
            os.makedirs(output_folder, exist_ok=True)
            try:
                nifti_to_png_slices_binary(label_path, output_folder)
            except:
                print('[ERROR]:', filename)


def main(mri_3d_img_tr_dir, mri_3d_label_tr_dir, mri_3d_img_ts_dir, mri_3d_label_ts_dir, mri_2d_img_tr_dir, mri_2d_label_tr_dir, mri_2d_img_ts_dir, mri_2d_label_ts_dir):

    os.makedirs(mri_2d_img_tr_dir, exist_ok=True)
    os.makedirs(mri_2d_label_tr_dir, exist_ok=True)
    os.makedirs(mri_2d_img_ts_dir, exist_ok=True)
    os.makedirs(mri_2d_label_ts_dir, exist_ok=True)

    mri_3d_2d_slice(mri_3d_img_tr_dir, mri_2d_img_tr_dir)
    label_3d_2d_slice(mri_3d_label_tr_dir, mri_2d_label_tr_dir)

    mri_3d_2d_slice(mri_3d_img_ts_dir, mri_2d_img_ts_dir)
    label_3d_2d_slice(mri_3d_label_ts_dir, mri_2d_label_ts_dir)

