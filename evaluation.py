import os
import lib.utils as utils
import skimage.measure
import numpy as np
import nibabel as nib
import glob
import SimpleITK as sitk
import csv
from collections import defaultdict

smooth = 0.001


def path2np(data_path):
    mri = nib.load(data_path)
    mri_np = np.asarray(mri.get_fdata(dtype=np.float32))
    mri_affine = mri.affine
    return mri_np, mri_affine


def calculate_connect_component(data_np):
    # calculate the number of connect components in the MRI data
    _, num_connect_component = skimage.measure.label(data_np, return_num=True)
    return num_connect_component


def evaluate_connected_component_3d(predicted_image, label_image):
    # calculate the number of connect components in the predicted image and the label image
    pred_cc = calculate_connect_component(predicted_image)
    label_cc = calculate_connect_component(label_image)
    # return the absolute difference of the number of connect components between the predicted image and the label image
    return abs(label_cc - pred_cc)


def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def hausdorff_distance_95(y_true, y_pred):
    # Convert arrays to SimpleITK images
    label_pred = sitk.GetImageFromArray(y_pred, isVector=False)
    label_true = sitk.GetImageFromArray(y_true, isVector=False)

    # Generate signed Maurer distance maps
    signed_distance_map_true = sitk.SignedMaurerDistanceMap(label_true > 0.5, squaredDistance=False,
                                                            useImageSpacing=True)
    signed_distance_map_pred = sitk.SignedMaurerDistanceMap(label_pred > 0.5, squaredDistance=False,
                                                            useImageSpacing=True)

    # Generate absolute distance maps
    ref_distance_map = sitk.Abs(signed_distance_map_true)
    seg_distance_map = sitk.Abs(signed_distance_map_pred)

    # Generate label contours
    ref_surface = sitk.LabelContour(label_true > 0.5, fullyConnected=True)
    seg_surface = sitk.LabelContour(label_pred > 0.5, fullyConnected=True)

    # Calculate distances from one surface to the other
    seg2ref_distance_map = ref_distance_map * sitk.Cast(seg_surface, sitk.sitkFloat32)
    ref2seg_distance_map = seg_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    # Calculate the number of surface pixels
    num_ref_surface_pixels = int(sitk.GetArrayViewFromImage(ref_surface).sum())
    num_seg_surface_pixels = int(sitk.GetArrayViewFromImage(seg_surface).sum())

    # Convert SimpleITK images to numpy arrays
    seg2ref_distance_map_arr = sitk.GetArrayFromImage(seg2ref_distance_map)
    ref2seg_distance_map_arr = sitk.GetArrayFromImage(ref2seg_distance_map)

    # Get all non-zero distances (those are the distances from one surface to the other)
    seg2ref_distances = seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0]
    ref2seg_distances = ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0]

    # Add zeros to the list of distances until its length is equal to the number of pixels in the contour
    seg2ref_distances = np.concatenate([seg2ref_distances, np.zeros(num_seg_surface_pixels - len(seg2ref_distances))])
    ref2seg_distances = np.concatenate([ref2seg_distances, np.zeros(num_ref_surface_pixels - len(ref2seg_distances))])

    # Calculate the union of both distance sets
    all_surface_distances = np.concatenate([seg2ref_distances, ref2seg_distances])

    # Calculate HD95
    hd95 = np.percentile(all_surface_distances, 95)

    return hd95


def bootstrap_ci(data, statistic=np.mean, alpha=0.05, num_samples=5000):
    n = len(data)
    rng = np.random.RandomState(47)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    stat = np.sort(statistic(samples, axis=1))
    lower = stat[int(alpha / 2 * num_samples)]
    upper = stat[int((1 - alpha / 2) * num_samples)]
    return lower, upper


def cal_avg_bootstrap_confidence_interval(x):
    x_avg = np.average(x)
    bootstrap_ci_result = bootstrap_ci(x)
    return np.round(x_avg, 4), np.round(bootstrap_ci_result[0], 4), np.round(bootstrap_ci_result[1], 4)


def evaluation_dicts2csv(filename_list, metric_names, metrics_result_list, csv_file_path):
    metric_names = ['filename'] + metric_names

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metric_names)

        writer.writeheader()
        for filename, metrics_result in zip(filename_list, metrics_result_list):
            metrics_result['filename'] = filename
            writer.writerow(metrics_result)


def average_eval_bci(dicts):
    averaged_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            # Skip non-numerical values
            if isinstance(value, (int, float)):
                averaged_dict[key].append(value)
    # Calculate average and bootstrap confidence interval
    averaged_dict = {key: cal_avg_bootstrap_confidence_interval(values) for key, values in averaged_dict.items()}
    return averaged_dict


def dict2str(evaluation_metrics, dicts):
    output = ', '.join([f"{key}: {dicts[key]}" for key in evaluation_metrics])
    return output


def pred_eval(pred_base_path, label_base_path, evaluation_metrics, evaluation_funcs, csv_file_path):
    assert len(evaluation_metrics) == len(evaluation_funcs)
    pred_path_list = glob.glob(f'{pred_base_path}/*.nii.gz')
    filename_list = []
    evaluation_metrics_result_list = []
    for pred_path in pred_path_list:
        filename = os.path.basename(pred_path)
        label_path = f'{label_base_path}/{filename}'
        pred_np, _ = path2np(pred_path)
        label_np, _ = path2np(label_path)
        try:
            evaluation_matrix_result = {metric: func(label_np, pred_np) for metric, func in
                                        zip(evaluation_metrics, evaluation_funcs)}
            evaluation_matrix_result_rounded = {key: round(value, 4) for key, value in evaluation_matrix_result.items()}
            print(f'{filename}: {dict2str(evaluation_metrics, evaluation_matrix_result_rounded)}')
        except Exception as e:
            print('[ERROR]', filename, e)
        filename_list.append(filename)
        evaluation_metrics_result_list.append(evaluation_matrix_result)
    evaluation_dicts2csv(filename_list, evaluation_metrics, evaluation_metrics_result_list, csv_file_path)
    evaluation_metrics_result_avg = average_eval_bci(evaluation_metrics_result_list)
    print('Average:', evaluation_metrics_result_avg)


def main():
    pred_base_path = 'your/prediction/path'
    label_base_path = 'your/label/path'
    evaluation_metrics = ['dice', 'hausdorff_95', 'topo_3d_err']
    evaluation_funcs = [dice_score, hausdorff_distance_95, evaluate_connected_component_3d]
    csv_file_path = f'{os.path.dirname(pred_base_path)}/eval_{os.path.basename(pred_base_path)}.csv'
    print('Saved to:', csv_file_path)
    pred_eval(pred_base_path, label_base_path, evaluation_metrics, evaluation_funcs, csv_file_path)


if __name__ == '__main__':
    main()
