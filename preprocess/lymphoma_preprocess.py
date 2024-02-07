import glob
import os
import torchio as tio


def data_transforms():
    resampling = tio.Resample(target=1)
    to_canonical = tio.ToCanonical()
    copy_affine = tio.CopyAffine('image')
    rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5))
    # Choose between CropOrPad or Resize
    croppad = tio.CropOrPad(target_shape=(240, 240, 160))
    # resize = tio.Resize(target_shape=(240, 240, 160))
    transforms_comb = tio.Compose([resampling, copy_affine, croppad, to_canonical, rescale])
    return transforms_comb


def path2tiosubject(data_path, label_path):
    tio_img = tio.ScalarImage(data_path)
    label = tio.LabelMap(label_path)
    name = os.path.basename(data_path)
    resampling = tio.Resample(tio_img)
    label_resampled = resampling(label)
    tio_subj = tio.Subject(image=tio_img, label=label_resampled, name=name)
    return tio_subj


def main():
    mri_dir = 'your/t1/weighted/lymphoma/MRI'
    label_dir = 'your/t1/weighted/lymphoma/label'
    mri_processed_save_dir = 'your/t1/weighted/lymphoma/process/MRI'
    label_processed_save_dir = 'your/t1/weighted/lymphoma/process/label'
    os.makedirs(mri_processed_save_dir, exist_ok=True)
    os.makedirs(label_processed_save_dir, exist_ok=True)
    transform = data_transforms()
    mri_path_list = glob.glob(f'{mri_dir}/*.nii.gz')
    for mri_path in mri_path_list:
        mri_name = os.path.basename(mri_path)
        label_path = f'{label_dir}/{mri_name}'
        tio_subject = path2tiosubject(mri_path, label_path)
        tio_subject_resample = transform(tio_subject)
        tio_subject_resample['image'].save(f'{mri_processed_save_dir}/{mri_name}')
        tio_subject_resample['label'].save(f'{label_processed_save_dir}/{mri_name}')


if __name__ == '__main__':
    main()
