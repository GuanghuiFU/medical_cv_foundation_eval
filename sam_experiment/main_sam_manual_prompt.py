import sam_experiment.sam_pred_3d_box as sam_pred_3d_box

image_dir = 'your/base/path/FM_experiments/imagesTs'
box_prompt_manual_3d_path = 'brats_prompt/3d_manual.csv'
pred_save_dir = f'your/base/path/FM_experiments/experiment_sam'

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
pred_save_path_3d_manual = f'{pred_save_dir}/3d_manual/{sam_model_name}'

# 1. Inference SAM
# SAM inference using 3D level manually labeled box prompt
sam_pred_3d_box.main(sam_model_name, sam_checkpoint_path, image_dir, box_prompt_manual_3d_path, pred_save_path_3d_manual)
