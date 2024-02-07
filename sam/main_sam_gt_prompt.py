import os
import label2box_2d
import label2box_3d
import sam_pred_2d_box
import sam_pred_3d_box

image_dir = 'your/train/image/path/imagesTr'
label_dir = 'your/train/image/path/labelsTr'
box_prompt_gt_2d_folder_path = 'brats_prompt/2d_gt'
box_prompt_gt_3d_path = 'brats_prompt/3d_gt.csv'
pred_save_dir = f'your/prediction/save/path'

# SAM related setting
sam_model_name_list = ['sam_vit_b_01ec64','sam_vit_l_0b3195','sam_vit_h_4b8939','medsam_vit_b']
sam_model_name = sam_model_name_list[0]
sam_checkpoint_path = f"checkpoints/{sam_model_name}.pth"

# 1. Inference SAM
# SAM inference using box prompts generated from 2D slices of ground truth
pred_save_path_2d_gt = f'{pred_save_dir}/{os.path.basename(box_prompt_gt_2d_folder_path)}/{sam_model_name}'
sam_pred_2d_box.main(sam_model_name, sam_checkpoint_path, image_dir, box_prompt_gt_2d_folder_path, pred_save_path_2d_gt)

# SAM inference using box prompts generated from 3D volumes of ground truth
pred_save_path_3d_gt = f'{pred_save_dir}/{os.path.basename(box_prompt_gt_3d_path)}/{sam_model_name}'
sam_pred_3d_box.main(sam_model_name, sam_checkpoint_path, image_dir, box_prompt_gt_3d_path, pred_save_path_3d_gt)

# 2. Create prompt from ground truth (if you want to try)
label2box_2d.main(label_dir, box_prompt_gt_2d_folder_path)
label2box_3d.main(label_dir, box_prompt_gt_3d_path)
