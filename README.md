# 3MT-RoadSeg
## Multi-modal Multi-task Road Segmentation

This repository contains the implementation of the 3MT-RoadSeg method in Pytorch. 3MT-RoadSeg is a fast and accurate method that does not need any preprocessing and uses only raw sensor inputs.

The training setup in the repo is now available for the KITTI and Cityscapes datasets. Training setups will be created for new data sets in the coming period.

![Resim2](https://user-images.githubusercontent.com/50530899/215703789-ed633586-ca8c-4d44-a74d-366e201a3cd5.png)


### Training
The configuration files to train the model can be found in the `configs/` directory. The model can be trained by running the following command:

```shell
python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
```

### Evaluation
We evaluate the best model at the end of training. The evaluation criterion is based on Equation 10 from our survey paper and requires to pre-train a set of single-tasking networks beforehand. To speed-up training, it is possible to evaluate the model only during the final 10 epochs by adding the following line to your config file:

```python
eval_final_10_epochs_only: True
``` 

## Support
The following datasets and tasks are supported.

| Dataset      | Sem. Seg. | Depth | Normals |
| KITTI        |     Y     |   Y   |    Aux  | 
| Cityscapes   |     Y     |   Aux |    Y    | 


The following models are supported.

| Backbone | HRNet | ResNet |
|----------|----------|-----------|
| Single-Task |  Y    |  Y |
| Multi-Task | Y | Y |
| Cross-Stitch | | Y |
| NDDR-CNN | | Y |
| MTAN | | Y |
| PAD-Net | Y | |
| MTI-Net | Y | |





