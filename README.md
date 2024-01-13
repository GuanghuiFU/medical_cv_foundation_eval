# Comparative analysis of supervised learning and computer vision foundation models for segmenting post-contrast T1-weighted primary brain lymphoma

* Conference version:

## Introduction

This repository presents our work on evaluating supervised learning and foundation models in medical imaging, specifically using T1-weighted brain MRI data. Our approach involves training supervised learning models on public datasets for glioma and comparing their performance against foundation models. The highlight of our study is the utilization of 3D supervised learning models, which have demonstrated superior performance in handling tasks with high variability in clinical data.

Key findings from our research indicate that while foundation models show potential, they currently do not match the effectiveness of 3D supervised learning models in complex medical imaging tasks. For example, our 3D model outperformed the best foundation model by approximately 20% in Dice score for the clinical dataset and showed a 10% improvement on public glioma datasets.

This code repository offers an implementation of our 3D supervised learning approach, with example code based on the BRATS brain tumor dataset. While our primary dataset involving lymphoma is not included due to privacy considerations, the methods and findings are broadly applicable to similar medical imaging tasks. 

## Manual Box Prompt annotation

The figure below shows our process of using ITK-SNAP for box annotation. Our purpose in using this tool is to maintain consistency with physician annotations.

![manual_box](https://github.com/GuanghuiFU/medical_cv_foundation_eval/blob/main/manual_box_prompt.png)

## Code explanation

* `sam_pred.py`: it starts by loading a 3D-level box prompt from an Excel file. The core process involves predicting on axial level slices using the SAM [1]. Key steps include:
  * **3D-to-2D Conversion:** Transforms the 3D level box prompt into a 2D format for SAM processing.
  * **SAM Prediction:** Utilizes SAM-based prediction on each 2D axial slice.
  * **2D-to-3D Reconstruction:** After processing slices, it reconstructs the 2D predictions back into a 3D volume, enabling performance evaluation.
* `universeg_select_support_set.py`: this code offers strategies for selecting the support set, a crucial step in preparing data for the UniverSeg [2] model. It includes options to select slices based on their size (largest, smallest, or medium). 
* `universeg_pred.py`: this is for segmenting 3D brain MRI data using the UniverSeg model. The process involves several transformation and processing steps:
  * **3D to 2D Transformation:** Converts each slice from the 3D volume into a 2D image in axial plane.
  * **Resizing for UniverSeg:** Resizes these 2D slices from their original resolution to a uniform size of 128x128 pixels.
  * **Support Set Configuration:** Requires setting up a support set for UniverSeg.
  * **Slice-by-Slice Processing:** Applies the UniverSeg model to each 2D slice in the axial plane.
  * **Reconstruction and Evaluation:** Collects the 2D predictions, resizes them back to their original resolution, and reconstructs them into a 3D volume for performance evaluation.

## References

1. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson,
   Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything.
   In Proc. ICCV 2023, pages 4015–4026, 2023
2. Victor Ion Butoi, Jose Javier Gonzalez Ortiz, Tianyu Ma, Mert R Sabuncu, John Guttag,
   and Adrian V Dalca. UniverSeg: Universal medical image segmentation. In Proc. ICCV
   2023, pages 21438–21451, 2023.
3. Michela Antonelli, Annika Reinke, Spyridon Bakas, Keyvan Farahani, Annette Kopp-
   Schneider, Bennett A Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger,
   Ronald M Summers, et al. The medical segmentation decathlon. Nature communications,
   13(1):4128, 2022.
4. Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein.
   nnU-Net: a self-configuring method for deep learning-based biomedical image segmenta-
   tion. Nature methods, 18(2):203–211, 2021.

## Citing us

* Fu, Guanghui, Rosana El Jurdi, Lydia Chougar, Didier Dormont, Romain Valabregue, Stéphane Lehéricy, and Olivier Colliot. "Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss." In *SPIE Medical Imaging 2023*. 2023.

```
@inproceedings{fu2023introducing,
  title={Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss},
  author={Fu, Guanghui and El Jurdi, Rosana and Chougar, Lydia and Dormont, Didier and Valabregue, Romain and Leh{\'e}ricy, St{\'e}phane and Colliot, Olivier},
  booktitle={SPIE Medical Imaging 2023},
  year={2023}
}
```
