## 3MT-RoadSeg
### Multi-modal Multi-task Road Segmentation

This repository contains the implementation of the 3MT-RoadSeg method in Pytorch. 3MT-RoadSeg is a fast and accurate method that does not need any preprocessing and uses only raw sensor inputs.

The training setup in the repo is now available for the KITTI and Cityscapes datasets. Training setups will be created for new data sets in the coming period.

![Resim2](https://user-images.githubusercontent.com/50530899/215703789-ed633586-ca8c-4d44-a74d-366e201a3cd5.png)

## Package Versions
The package versions used during code development are as follows:
- Python: 3.7
- Pytorch: 1.9.0
- Cudatoolkit: 10.2
- Torchvision: 0.10.0
- Opencv-python: 4.5.3.56

## Datasets
You can download the KITTI road dataset from [KITTI](https://www.cvlibs.net/datasets/kitti/) and the Cityscapes road dataset from [Cityscapes](https://www.cityscapes-dataset.com/). Depth images can be obtained from the [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg). You can access the 3-channel ADIs from here.

```
3MT-RoadSeg
 |-- data
 |  |-- to
 |  |  |-- databases
 |  |  |  |-- KITTIRoad
 |  |  |  |  |-- training
 |  |  |  |  |  |-- ...
 |  |  |  |  | -- testing
 |  |  |  |  |  |-- ...
 |  |  |  |-- CityScapes
 |  |  |  |  |-- training
 |  |  |  |  |  |-- ...
 |  |  |  |  |-- testing
 |  |  |  |  |  |-- ...
 ...
```


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

## References
This repository has been built upon the foundation of repository [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).


