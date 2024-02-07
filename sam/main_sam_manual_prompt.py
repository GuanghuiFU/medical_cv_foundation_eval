import os
import sam_pred_3d_box

image_dir = 'your/train/image/path/imagesTr'
box_prompt_manual_3d_path = 'brats_prompt/3d_manual.csv'
pred_save_dir = f'your/prediction/save/path'

# SAM related setting
sam_model_name_list = ['sam_vit_b_01ec64','sam_vit_l_0b3195','sam_vit_h_4b8939','medsam_vit_b']
sam_model_name = sam_model_name_list[0]
sam_checkpoint_path = f"checkpoints/{sam_model_name}.pth"
pred_save_path_3d_manual = f'{pred_save_dir}/{os.path.basename(box_prompt_manual_3d_path)}/{sam_model_name}'

# 1. Inference SAM
# SAM inference using 3D level manually labeled box prompt
sam_pred_3d_box.main(sam_model_name, image_dir, box_prompt_manual_3d_path, pred_save_path_3d_manual, pred_save_path_3d_manual)