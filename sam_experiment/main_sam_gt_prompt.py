import os
import sam_experiment.label2box_2d as label2box_2d
import sam_experiment.label2box_3d as label2box_3d
import sam_experiment.sam_pred_2d_box as sam_pred_2d_box
import sam_experiment.sam_pred_3d_box as sam_pred_3d_box

image_dir = 'C:/Users/fugua/Downloads/FM_experiments/imagesTs'
label_dir = 'C:/Users/fugua/Downloads/FM_experiments/labelsTs'
box_prompt_gt_2d_folder_path = 'brats_prompt/2d_gt'
box_prompt_gt_3d_path = 'brats_prompt/3d_gt.csv'
pred_save_dir = "C:/Users/fugua/Downloads/FM_experiments/experiment_sam"
os.makedirs(pred_save_dir, exist_ok=True)

# SAM related setting
sam_model_name_list = ['sam_vit_b_01ec64','sam_vit_l_0b3195','sam_vit_h_4b8939','medsam_vit_b']
'''
Install SAM repository: pip install git+https://github.com/facebookresearch/segment-anything.git

Download SAM models from the following paths:
1. SAM, vit-b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
2. SAM, vit-h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
3. SAM, vit-l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
4. MedSAM, vit-b: https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view?usp=drive_link
And put them in the checkpoints path.
'''
sam_model_name = sam_model_name_list[0]
sam_checkpoint_path = f"checkpoints/{sam_model_name}.pth"

# 1. Inference SAM
# SAM inference using box prompts generated from 2D slices of ground truth
print('***Inferencing for SAM 2d-gt prompt experiment...')
pred_save_path_2d_gt = f'{pred_save_dir}/2d_gt/{sam_model_name}'
sam_pred_2d_box.main(sam_model_name, sam_checkpoint_path, image_dir, box_prompt_gt_2d_folder_path, pred_save_path_2d_gt)

# SAM inference using box prompts generated from 3D volumes of ground truth
print('***Inferencing for SAM 3d-gt prompt experiment...')
pred_save_path_3d_gt = f'{pred_save_dir}/3d_gt/{sam_model_name}'
sam_pred_3d_box.main(sam_model_name, sam_checkpoint_path, image_dir, box_prompt_gt_3d_path, pred_save_path_3d_gt)

# 2. Create prompt from ground truth (if you want to try)
# label2box_2d.main(label_dir, box_prompt_gt_2d_folder_path)
# label2box_3d.main(label_dir, box_prompt_gt_3d_path)
