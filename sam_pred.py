import numpy as np
import pandas as pd
import nibabel as nib
from segment_anything import sam_model_registry, SamPredictor
import torch
import glob
import os
import time


def get_position_size(filename, excel_path):
    df = pd.read_csv(excel_path)
    row = df[df['Filename'] == filename]
    if not row.empty:
        position = row['Position'].values[0]
        size = row['Size'].values[0]
        return position, size
    else:
        return None, None


def get_2d_annotations_from_3d_box(position, size):
    x, y, z = position
    dx, dy, dz = size
    # Calculate the range of slices in the Z-axis that intersect the box
    z_start = z
    z_end = z + dz
    # List to store 2D annotations
    annotations_box_dict_list = []
    # Iterate over each slice in the Z-axis
    for slice_z in range(z_start, z_end):
        # Each 2D annotation is a rectangle with the same x, y, and size in x, y
        annotation_box = [[y, x, y + dy, x + dx]]
        ## In the visualization, I found the operation after load from nifti to png, the position changed, horizontal flip and then rotate 90 degree. I checked from MRview, found that the Coordinate is at right bottom
        # annotation_box = [[x, x + dx, y, y + dy]]
        annotations_box_dict_list.append({"slice": slice_z, "box": annotation_box})
    return annotations_box_dict_list


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


def sam_predict(sam_model_name, sam_type, mri_data, annotations_box_3d_list):
    prediction = np.zeros_like(mri_data)
    sam_checkpoint = f"checkpoints/{sam_model_name}.pth"
    device = "cuda"
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for annotations_box_3d in annotations_box_3d_list:
        slice = int(annotations_box_3d['slice'])
        box = annotations_box_3d['box']
        mri_slice_8bit = (mri_data[:, :, slice] * 255).astype(np.uint8)
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
        prediction[:, :, slice] = masks[0].cpu()
    return prediction

# The whole folder
def main():
    sam_model_dict = {'sam_vit_b_01ec64': 'vit_b',
                      'sam_vit_l_0b3195': 'vit_l',
                      'sam_vit_h_4b8939': 'vit_h',
                      'medsam_vit_b': 'vit_b'}
    for sam_model_name, sam_model_type in sam_model_dict.items():
        print(sam_model_name, sam_model_type)

        mri_dir = 'your/data/path'
        save_dir = f'your/prediction/save/path/{sam_model_name}'
        os.makedirs(save_dir, exist_ok=True)
        excel_path = 'your/box/prompt/path/brats_box_prompt.csv'
        mri_path_list = glob.glob(f'{mri_dir}/*.nii.gz')
        process_times = []
        for mri_path in mri_path_list:
            try:
                mri_name = os.path.basename(mri_path)
                print('Processing:', mri_name, '. mri_path:', mri_path)
                start_time = time.time()
                position, size = get_position_size(mri_name, excel_path)
                print('Position:', position, ', Size:', size)
                if position is not None:
                    position = [int(item) for item in position.split(',')]
                    size = [int(item) for item in size.split(',')]
                    annotation_list = get_2d_annotations_from_3d_box(position, size)
                    mri_data = normalize_image(load_nifti_data(mri_path))
                    prediction = sam_predict(sam_model_name, sam_model_type, mri_data, annotation_list)
                    save_nifti_data(prediction, mri_path, f'{save_dir}/{mri_name}')
                    processing_time = time.time() - start_time
                    print('Time:', processing_time, 'seconds')
                    process_times.append({'mri_name': mri_name, 'processing_time': processing_time})
                process_times_csv = pd.DataFrame(process_times)
                process_times_csv.to_csv(f'processing_times_{sam_model_type}.csv', index=False)
            except Exception as e:
                print("[ERROR]:", mri_name, e)

if __name__ == '__main__':
    main()
