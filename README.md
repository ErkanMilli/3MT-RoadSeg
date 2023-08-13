# UNDER CONSTRUCTION...

## 3MT-RoadSeg
### Multi-modal Multi-task Road Segmentation

This repository contains the implementation of the 3MT-RoadSeg method in Pytorch. 3MT-RoadSeg is a fast and accurate method that does not need any preprocessing and uses only raw sensor inputs.

The training setup in the repo is now available for the KITTI and Cityscapes datasets. Training setups will be created for new data sets in the coming period.

![Resim2](https://user-images.githubusercontent.com/50530899/215703789-ed633586-ca8c-4d44-a74d-366e201a3cd5.png)




## Support
The following datasets and tasks are supported.

| Dataset      | Segmentation | Depth | Normals |
|--------------|-----------|-------|---------|
| KITTI        |     Y     |   Y   |    Aux  | 
| Cityscapes   |     Y     |   Aux |    Y    | 


The following models are supported.

| Backbone | HRNet-w18 | HRNet-w32 | HRNet-w48 |
|----------|-----------|-----------|-----------|
| 3MT-RoadSeg |  Y  |  Y  |  Y  |





