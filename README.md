# CRFSilhouette (Master Thesis)
Silhouette extraction using Mask-RCNN and Postprocess with dense CRF

This repository contains the code for independently segmenting the dynamic object of interest. This is the first part of the proposed pipeline. 

Deep Learning based pretrained instance segmentation model MaskRCNN was used to detect and extract the silhouette of the object of interest. Any object on which the model is trained
on can be used to extract silhouette. In this case, since the datasets used in this research are all "people" as the dynamic object, the code is written to segment and extract "people". 
It was also found that this can also work with multiple instances as well.

The segmentations from MaskRCNN are then passed to a post process method that uses Fully Connected Condition Random Field (FC-CRF) to refine the silhouettes.

Future work:
Incorporating transfer learning on the existing pretrained model will improve the accuracy and the quality of final reconstructions. 
