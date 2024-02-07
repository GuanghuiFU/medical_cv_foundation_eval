import os
import universeg_data_process
import universeg_select_support_set
import universeg_pred

data_split_csv_path = '../brats-data_split.csv'

mri_3d_img_tr_dir = 'your/train/image/path/imagesTr'
mri_3d_label_tr_dir = 'your/train/label/path/labelsTr'
mri_3d_img_ts_dir = 'your/test/image/path/imagesTs'
mri_3d_label_ts_dir = 'your/test/label/path/labelsTs'

mri_2d_img_tr_dir = 'your/train/image/path/slice_128_128/imagesTr_slice'
mri_2d_label_tr_dir = 'your/train/label/path/slice_128_128/labelsTr_slice'
mri_2d_img_ts_dir = 'your/test/image/path/slice_128_128/imagesTs_slice'
mri_2d_label_ts_dir = 'your/test/label/path/slice_128_128/labelsTs_slice'

support_set_dir = 'your/support/set/path'
selection = 'largest' # Choose between smallest, middle, largest

pred_save_dir = f'your/prediction/save/path_{os.path.basename(support_set_dir)}'

# 1. Process your data
universeg_data_process.main(mri_3d_img_tr_dir, mri_3d_label_tr_dir, mri_3d_img_ts_dir, mri_3d_label_ts_dir,
                            mri_2d_img_tr_dir, mri_2d_label_tr_dir, mri_2d_img_ts_dir, mri_2d_label_ts_dir)
# 2. Create support set
universeg_select_support_set.main(data_split_csv_path, mri_3d_label_tr_dir, support_set_dir, selection)

# 3. Inference the UniverSeg model
universeg_pred.main(mri_2d_img_ts_dir, support_set_dir, selection, pred_save_dir, 0.9)