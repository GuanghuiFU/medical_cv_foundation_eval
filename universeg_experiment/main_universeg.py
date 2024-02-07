import universeg_experiment.universeg_data_process as universeg_data_process
import universeg_experiment.universeg_select_support_set as universeg_select_support_set
import universeg_experiment.universeg_pred as universeg_pred

data_split_csv_path = '../brats-data_split.csv'

mri_3d_img_tr_dir = 'C:/Users/fugua/Downloads/FM_experiments/imagesTr'
mri_3d_label_tr_dir = 'C:/Users/fugua/Downloads/FM_experiments/labelsTr'
mri_3d_img_ts_dir = 'C:/Users/fugua/Downloads/FM_experiments/imagesTs'
mri_3d_label_ts_dir = 'C:/Users/fugua/Downloads/FM_experiments/labelsTs'

mri_2d_img_tr_dir = 'C:/Users/fugua/Downloads/FM_experiments/slice_128_128/imagesTr_slice'
mri_2d_label_tr_dir = 'C:/Users/fugua/Downloads/FM_experiments/slice_128_128/labelsTr_slice'
mri_2d_img_ts_dir = 'C:/Users/fugua/Downloads/FM_experiments/slice_128_128/imagesTs_slice'
mri_2d_label_ts_dir = 'C:/Users/fugua/Downloads/FM_experiments/slice_128_128/labelsTs_slice'

support_set_dir = 'C:/Users/fugua/Downloads/FM_experiments/slice_128_128/support_set'
selection_list = ['smallest', 'middle', 'largest']
selection = selection_list[2]
threshold = 0.9
pred_save_dir = f'C:/Users/fugua/Downloads/FM_experiments/experiment_universeg/{selection}_{threshold}'

# Install UniverSeg repository: pip install git+https://github.com/JJGO/UniverSeg.git

#1. Process your data
print('***Transfer data from 3d to 2D and resize to (128,128)...')
universeg_data_process.main(mri_3d_img_tr_dir, mri_3d_label_tr_dir, mri_3d_img_ts_dir, mri_3d_label_ts_dir,
                            mri_2d_img_tr_dir, mri_2d_label_tr_dir, mri_2d_img_ts_dir, mri_2d_label_ts_dir)
#2. Create support set
print(f'***Select support set based on the {selection} area in the volume...')
universeg_select_support_set.main(data_split_csv_path, mri_2d_label_tr_dir, support_set_dir, selection)

# 3. Inference the UniverSeg model
print(f'***Inferencing UniverSeg based on support set ({selection})')
universeg_pred.main(mri_2d_img_ts_dir, f'{support_set_dir}_{selection}', pred_save_dir, threshold)