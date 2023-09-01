## 3MT-RoadSeg
### Multi-modal Multi-task Road Segmentation

This repository contains the implementation of the 3MT-RoadSeg method in Pytorch. 3MT-RoadSeg is a fast and accurate method that does not need any preprocessing and uses only raw sensor inputs.

The training setup in the repo is now available for the KITTI and Cityscapes datasets. Training setups will be created for new data sets in the coming period.

![Scheme](https://user-images.githubusercontent.com/50530899/215703789-ed633586-ca8c-4d44-a74d-366e201a3cd5.png)

![3MT_ss](https://github.com/ErkanMilli/3MT-RoadSeg/blob/main/3MT_ss.png = 350x250)

## Package Versions
The package versions used during code development are as follows:
- Python: 3.7
- Pytorch: 1.9.0
- Cudatoolkit: 10.2
- Torchvision: 0.10.0
- Opencv-python: 4.5.3.56

## Datasets
You can download the KITTI road dataset from [KITTI](https://www.cvlibs.net/datasets/kitti/) and the Cityscapes road dataset from [Cityscapes](https://www.cityscapes-dataset.com/). Depth images can be obtained from the [SNE-RoadSeg](https://github.com/hlwang1124/SNE-RoadSeg). You can access the 3-channel ADIs from [here](https://drive.google.com/drive/folders/1n3CgKbr3OgfZ7YYE-dX5JrCHjmcDnbY5?usp=drive_link).

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
Configuration files are located in the ```configs/``` directory. You have the option to alter the training settings from this directory if you prefer. You can split the original training set into a new training set and a validation set as you wish. Then run the script below:
```
train.py --config_env configs/env.yml --config_exp configs/$DATASET/$BACKBONE/$MODEL.yml
```
You can perform a test on the validation set created by splitting the training set. You will need the chechpoint.pth.rar file for this. 
```
test.py ---config_env configs/env.yml --config_exp configs/$DATASET/$BACKBONE/$MODEL.yml
```
Additionally, if you only want to output for single input, you can do this using test_singleInput.py like this: 
```
test_singleInput.py ---config_env configs/env.yml --config_exp configs/$DATASET/$BACKBONE/$MODEL.yml
```

You can use the same split we used for the validation set generated from the training set. The train & test split is as follows:
```
|                  Train Split                   |              |                   Test Split                   |
|------------------------------------------------|              |------------------------------------------------|
um_000000.png  |  umm_000000.png  |  uu_000000.png              um_000068.png  |  umm_000068.png  |  uu_000068.png
...            |  ...             |  ...                        ...            |  ...             |  ...
...            |  ...             |  ...                        ...            |  ...             |  ...
um_000067.png  |  umm_000067.png  |  uu_000067.png              um_000094.png  |  umm_000095.png  |  uu_000097.png

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

