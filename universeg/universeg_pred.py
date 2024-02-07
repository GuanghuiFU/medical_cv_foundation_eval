import glob
import os
from PIL import Image
from universeg import universeg
import itertools
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


def save_tensor_as_image(tensor, filename):
    # Convert the tensor to a NumPy array
    np_img = tensor.numpy()
    # Convert the array to an image
    img = Image.fromarray(np_img.astype("uint8"), mode="L")
    # Save the image
    img.save(filename)


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        label_path = os.path.join(self.label_dir, self.filenames[idx])
        image = torch.from_numpy(np.array(Image.open(img_path).convert('L')))
        label = torch.from_numpy(np.array(Image.open(label_path).convert('L')))
        image = min_max_normalize(image)
        label = min_max_normalize(label)
        return image, label


def load_and_resize_image(image_path, target_size=(240, 240)):
    img = Image.open(image_path).convert('L')
    img_resized = img.resize(target_size)
    img_np = np.array(img_resized)
    binarized_img = np.where(img_np > 127, 1, 0)
    return binarized_img


def png_to_nifti(pred_dir, subject_name):
    input_dir = os.path.join(pred_dir, subject_name)
    output_file = os.path.join(pred_dir, f"{subject_name}.nii.gz")
    mri_3d_label_dir = '/network/lustre/iss02/aramis/users/guanghui.fu/data/nnUNetFrame/DATASET/nnUNet_raw/Dataset912_BRATS/labelsTs'
    reference_nifti_path = f'{mri_3d_label_dir}/{subject_name}.nii.gz'
    reference_nifti = nib.load(reference_nifti_path)
    affine = reference_nifti.affine
    volume = np.zeros((240, 240, 155), dtype=np.uint8)
    for slice_index in range(155):
        file_name = f'slice_{slice_index}_image.png'
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            current_slice = load_and_resize_image(file_path)
        else:
            current_slice = np.zeros((240, 240), dtype=np.uint8)
        volume[:, :, slice_index] = current_slice
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_file)


def calculate_dice_coefficient(ground_truth_path, prediction_path):
    def dice_coefficient(binary_1, binary_2):
        intersection = np.sum(binary_1 * binary_2)
        size1 = np.sum(binary_1)
        size2 = np.sum(binary_2)
        dice = (2. * intersection) / (size1 + size2)
        return dice

    ground_truth_img = nib.load(ground_truth_path)
    prediction_img = nib.load(prediction_path)
    ground_truth_data = ground_truth_img.get_fdata()
    prediction_data = prediction_img.get_fdata()
    return dice_coefficient(ground_truth_data, prediction_data)


def get_support_set(support_set_dir, n_support, device="cuda"):
    support_img_dir = f'{support_set_dir}/images'
    support_label_dir = f'{support_set_dir}/labels'
    d_support = CustomDataset(support_img_dir, support_label_dir)
    support_images, support_labels = zip(*itertools.islice(d_support, n_support))
    print('len(support_images):', len(support_images))
    print('len(support_labels):', len(support_labels))
    support_images_tensor = torch.stack(support_images).unsqueeze(0).unsqueeze(2).to(device)  # [B, S, C, H, W]
    support_labels_tensor = torch.stack(support_labels).unsqueeze(0).unsqueeze(2).to(device)  # [B, S, 1]
    print('support_images_tensor.size():', support_images_tensor.size())
    print('support_labels_tensor.size():', support_labels_tensor.size())
    return support_images_tensor, support_labels_tensor


def universeg_process(model, subject_name, target_img_dir, support_images_tensor, support_labels_tensor, pred_save_dir,
                      device="cuda", threshold=0.7):
    target_img_dir = f'{target_img_dir}/{subject_name}'
    prediction_dir = f'{pred_save_dir}/{subject_name}'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    for img_file in os.listdir(target_img_dir):
        if img_file.endswith(('.png')):  # Assuming your images are in these formats
            target_img_path = os.path.join(target_img_dir, img_file)
            target_img = torch.from_numpy(np.array(Image.open(target_img_path).convert('L')))
            target_img = min_max_normalize(target_img)
            target_img = target_img.unsqueeze(0).unsqueeze(0).to(device)
            logits = model(target_img, support_images_tensor, support_labels_tensor)[0].to('cpu')
            pred = torch.sigmoid(logits)
            pred_binary = (pred > threshold).float().squeeze(0).squeeze(0)
            # save_image(pred_binary, os.path.join(prediction_dir, f'{img_file}'))
            save_tensor_as_image(pred_binary * 255, os.path.join(prediction_dir, f'{img_file}'))


def main(mri_2d_img_dir, support_set_base_dir, selection, pred_save_dir, threshold):
    device = 'cuda'
    support_set_dir = f'{support_set_base_dir}/{selection}'
    n_support = len(glob.glob(f'{support_set_dir}/*.png'))

    model = universeg(pretrained=True).to(device)
    test_subject_names = os.listdir(mri_2d_img_dir)
    for test_subject in tqdm(test_subject_names):
        print('Processing:', test_subject)
        support_images_tensor, support_labels_tensor = get_support_set(support_set_dir, n_support, device)
        universeg_process(model, test_subject, mri_2d_img_dir, support_images_tensor.to(device), support_labels_tensor.to(device), pred_save_dir, device, threshold)
        png_to_nifti(pred_save_dir, test_subject)
