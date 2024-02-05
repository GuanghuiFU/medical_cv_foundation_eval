# Comparing foundation models and nnU-Net for segmentation of primary brain lymphoma on clinical routine post-contrast T1-weighted MRI


This repository contains material associated to this [paper](#Citation).

It contains:
- link to trained models for segmentation of lymphoma from post-constrast T1-weighted MRI
- link to trained models for segmentation of enhancing tumor in MSD-BraTS datasets
- code and material for reproducing the experiments on MSD-BraTS

If you use this material, we would appreciate if you could cite the following reference.

## Citation
* Guanghui Fu, Lucia Nichelli, Dario Herran, Romain Valabregue, Agusti Alentorn, Khê Hoang-Xuan, Caroline Houillier, Didier Dormont, Stéphane Lehéricy, Olivier Colliot. Comparing foundation models and nnU-Net for segmentation of primary brain lymphoma on clinical routine post-contrast T1-weighted MRI. Submitted to *MIDL 2024*.

```
@inproceedings{fu2024comparing,
  title={Comparing foundation models and nnU-Net for segmentation of primary brain lymphoma on clinical routine post-contrast T1-weighted MRI},
  author={Fu, Guanghui and Nichelli, Lucia and Herran, Dario and Valabregue, Romain and Alentorn, Agusti and Hoang-Xuan, Khê and Houillier, Caroline and Dormont, Didier and Leh{\'e}ricy, St{\'e}phane and Colliot, Olivier},
  booktitle={Preprint},
  year={2024}
}
```
## Contents for reproducing MSD-BraTS experiments
We provide the following contents for reproduction of MSD-BraTS experiments:
- list of subjects of MSD-BraTS that were used ([link](<https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/brats-data_split.csv>))
- manual box prompts for SAM and MedSAM models ([link](#Manual-box-prompt-annotation))
- support sets for UniverSeg experiments ([link](#Support-sets-for-Universeg))
- code to train nnU-Net models ([link](#Code-to-train-nnU-net))
- code for inference of all models ([SAM](#Inference-for-SAM-models), [MedSAM](#Inference-for-MedSAM), [UniverSeg](#Inference-for-UniverSeg), [nnU-Net](#Inference-for-nnU-Net))
- code for computation of metrics and statistical analysis ([link](#Computation-of-metrics-and-statistical-analysis))

## Manual box prompt annotation

SAM and MedSAM require prompts. In this paper, we drew box prompts manually.

The figure below shows our process of using ITK-SNAP for drawing box prompts.

We also provide screen recording videos during annotation: https://owncloud.icm-institute.org/index.php/s/9LWatZ2xDB9SvE0

![manual_box](https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/manual_box_prompt.png)

In order to reproduce the experiments, you need the coordinates of the box prompts which are given here:
* `brats_3d_box_prompt_manual.csv`: The manual annotate box prompt in 3D level

## Generating boxes from grount truth
Also we provide the prompts generated from ground-truth.
* `brats_3d_box_prompt_label2box.csv`: The box prompt generate from ground truth in 3D level. 
* `brats_2d_box_prompt_label2box.csv`: The box prompt generate from ground truth in 2D level

The code to generate the boxes from ground truth is provided here:

* `label2box_roi.py` : The code to generate box prompt from ground truth in 3D level.
* `label2box_roi_2d_slice.py`: The code to generate box prompt from ground truth in 2D slice level.


## Support sets for Universeg

The different support sets are given here:  [`brats_support_set.zip`](<https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/brats_support_set.zip>) 

The code to build the support sets:
* `universeg_select_support_set.py`: this code offers strategies for selecting the support set, a crucial step in preparing data for the UniverSeg [2] model. It includes options to select slices based on their size (largest, smallest, or medium). 

## Code to train nnU-net

Training nnU-Net only requires the following commands: 

* **Plan and preprocessing**: nnUNetv2_plan_and_preprocess -d <ID> --verify_dataset_integrity
* **3D model training**: nnUNetv2_train <ID> 3d_fullres <cross validation fold id>
* **2D model training**: nnUNetv2_train <ID> 2d <cross validation fold id>

## Inference for SAM and MedSAM models

* `sam_pred.py`: it starts by loading a 3D-level box prompt from an Excel file. The core process involves predicting on axial level slices using the SAM [1]. Key steps include:
  * **3D-to-2D Conversion:** Transforms the 3D level box prompt into a 2D format for SAM processing.
  * **SAM Prediction:** Utilizes SAM-based prediction on each 2D axial slice.
  * **2D-to-3D Reconstruction:** After processing slices, it reconstructs the 2D predictions back into a 3D volume, enabling performance evaluation.

## Inference for UniverSeg
* `universeg_pred.py`: this is for segmenting 3D brain MRI data using the UniverSeg model. The process involves several transformation and processing steps:
  * **3D to 2D Transformation:** Converts each slice from the 3D volume into a 2D image in axial plane.
  * **Resizing for UniverSeg:** Resizes these 2D slices from their original resolution to a uniform size of 128x128 pixels.
  * **Support Set Configuration:** Requires setting up a support set for UniverSeg.
  * **Slice-by-Slice Processing:** Applies the UniverSeg model to each 2D slice in the axial plane.
  * **Reconstruction and Evaluation:** Collects the 2D predictions, resizes them back to their original resolution, and reconstructs them into a 3D volume for performance evaluation.

## Inference for nnU-Net

Inferencing nnU-Net only requires the following commands: 

* **3D inference**: nnUNetv2_predict -d <ID> -i <input path> -o <output path> -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans
* **2D inference**: nnUNetv2_predict -d <ID> -i <input path> -o <output path> -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans

## Computation of metrics and statistical analysis

* `evaluation.py`: This code is for evaluation and calculate the 95% bootstrap confidence interval ([link](<https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/evaluation.py>)).
* `t_test.py`: This code is to perform paired T-test ([link](<https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/t_test.py>)).


## Related codes

1. **SAM** [1]: https://github.com/bingogome/samm
2. **MedSAM** [5]: https://github.com/bowang-lab/MedSAM
3. **UniverSeg** [2]: https://github.com/JJGO/UniverSeg
4. **nnU-Net** [4]: https://github.com/MIC-DKFZ/nnUNet

## References

1. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proc. ICCV 2023, pages 4015–4026, 2023
2. Victor Ion Butoi, Jose Javier Gonzalez Ortiz, Tianyu Ma, Mert R Sabuncu, John Guttag, and Adrian V Dalca. UniverSeg: Universal medical image segmentation. In Proc. ICCV 2023, pages 21438–21451, 2023.
3. Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-Schneider, Bennett A Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M Summers, et al. The medical segmentation decathlon. Nature communications, 13(1):4128, 2022.
4. Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021.
5. Ma, Jun, et al. "Segment anything in medical images." *Nature Communications* 15.1 (2024): 654.



