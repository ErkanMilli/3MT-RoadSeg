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
## Usage
### Training
Configuration files are located in the ```configs/``` directory. You can split the original training set into a new training set and a validation set as you wish. Then run the script below:
```
training.py --config_env configs/env.yml --config_exp configs/$DATASET/$BACKBONE/$MODEL.yml
```
If you want to use the split we use as the validation set, you can download it here.

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

You can download the weights belonging to the pre-trained HRNet backbones from  [source](https://github.com/HRNet/HRNet-Image-Classification).

## References
This repository has been built upon the foundation of repository [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch).

## Citation
```
@ARTICLE{10182336,
  author={Milli, Erkan and Erkent, Özgür and Yılmaz, Asım Egemen},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Modal Multi-Task (3MT) Road Segmentation}, 
  year={2023},
  volume={8},
  number={9},
  pages={5408-5415},
  doi={10.1109/LRA.2023.3295254}}
```

