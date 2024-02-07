import numpy as np
import pandas as pd
import nibabel as nib
from segment_anything import sam_model_registry, SamPredictor
import torch
import glob
import os
import time


def get_annotations_from_csv(filename, prompt_csv_path):
    csv_file = f"{prompt_csv_path}/{filename.replace('.nii.gz','')}_slice_prompt.csv"
    df = pd.read_csv(csv_file)
    annotations_list = []
    for _, row in df.iterrows():
        slice_no = row['SliceNo']
        position = [int(item) for item in row['Position'].split(',')]
        size = [int(item) for item in row['Size'].split(',')]
        x, y = position
        dx, dy = size
        annotation_box = [[y, x, y + dy, x + dx]]
        annotations_list.append({"slice": slice_no, "box": annotation_box})
    return annotations_list


def normalize_image(image_data):
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    return (image_data - min_val) / (max_val - min_val)


def load_nifti_data(file_path):
    nifti_data = nib.load(file_path)
    data_array = nifti_data.get_fdata(dtype=np.float32)
    return data_array


def save_nifti_data(data_array, original_file_path, output_file_path):
    original_nifti = nib.load(original_file_path)
    new_nifti = nib.Nifti1Image(data_array, original_nifti.affine, original_nifti.header)
    nib.save(new_nifti, output_file_path)


def sam_predict(sam_type, sam_checkpoint_path, mri_data, annotations_list):
    prediction = np.zeros_like(mri_data)
    device = "cuda"
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for annotation in annotations_list:
        slice_no = int(annotation['slice'])
        box = annotation['box']
        mri_slice_8bit = (mri_data[:, :, slice_no] * 255).astype(np.uint8)
        mri_slice_8bit = np.repeat(mri_slice_8bit[..., np.newaxis], 3, axis=2)
        input_boxes = np.array(box)
        input_boxes = torch.from_numpy(input_boxes).to(predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, mri_slice_8bit.shape[:2])
        predictor.set_image(mri_slice_8bit)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        prediction[:, :, slice_no] = masks[0].cpu()
    return prediction



# The whole folder
def main(sam_model_name, sam_checkpoint_path, image_dir, prompt_csv_path, save_dir):
    sam_model_dict = {'sam_vit_b_01ec64': 'vit_b',
                      'sam_vit_l_0b3195': 'vit_l',
                      'sam_vit_h_4b8939': 'vit_h',
                      'medsam_vit_b': 'vit_b'}
    sam_model_type = sam_model_dict[sam_model_name]
    print(sam_model_name, sam_model_type, sam_checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)
    mri_path_list = glob.glob(f'{image_dir}/*.nii.gz')
    for mri_path in mri_path_list:
        mri_name = os.path.basename(mri_path)
        try:
            print('Processing:', mri_name, '. mri_path:', mri_path)
            annotations_list = get_annotations_from_csv(mri_name.replace("_0000",""), prompt_csv_path)
            mri_data = normalize_image(load_nifti_data(mri_path))
            prediction = sam_predict(sam_model_type, sam_checkpoint_path, mri_data, annotations_list)
            save_nifti_data(prediction, mri_path, f'{save_dir}/{mri_name}')
        except Exception as e:
            print("[ERROR]:", mri_name, e)

