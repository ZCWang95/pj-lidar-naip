---
---
---

# U-Net CNN for Woodland Canopy Height Mapping

## Summary

The two Python scripts in this repository (`unet_train.py` and `unet_predict.py`) were created to enable modeling and mapping (i.e., making spatially explicit predictions) of tree canopy height using a U-Net convolutional neural network (CNN). They were written in support of a manuscript, currently under review, that seeks to develop a robust, repeatable, and broadly applicable workflow for mapping woodland vegetation structure (canopy cover, aboveground biomass, and tree density).

## Background

Airborne lidar is the gold standard for vegetation structure mapping on broad spatial scales; however its spatiotemporal availability is limited. In the contiguous US, the National Agricultural Inventory Program (NAIP) collects high-resolution (i.e.., 0.3 - 0.6m) aerial imagery every few years on a statewide basis. CNNs are a widely used type of computer vision algorithm that leverages the color, shape, size, texture, and context information within image data to recognize patterns and use those patterns for predictive purposes. Indeed, CNNs have been shown recently to be useful for predicting vegetation height from high-resolution remotely sensed imagery.

## Objective

By training and validating a CNN model with a U-Net architecture on lidar-derived canopy heights (response variable) and NAIP aerial imagery (predictor data), we sought to map canopy height across diverse portions of the vast piñon-juniper (PJ) woodland ecosystem of the Southwestern US.

## Methods

We gathered lidar and NAIP data from 100 PJ-dominant sites, each 3x3 km in size, around the western US, and randomly split them into training (60), validation (20), and test (20). Lidar canopy height model (CHM) and NAIP imagery were split into 256x256 pixel tiles, with 16-pixel overlap, yielding 400 image chips for each site (24000 training, 8000 validation, and 8000 test chips in total). A U-Net model was built using the `tensorflow.keras` API (see details in `unet_train.py` and figure below).

\![U-Net Architecture](images/unet_fit.png)

## Results

Our models
