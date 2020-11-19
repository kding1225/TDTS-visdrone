# Train in Dense and Test in Sparse: A Method for Sparse Object Detection in Aerial Images

This project hosts the code for implementing the TDTS algorithm for object detection, as presented in our paper:

    Train in Dense, Test in Sparse: A Method for Sparse Object Detection in Aerial Images;
    Kun Ding, Guojin He, Huxiang Gu, Zisha Zhong, Shiming Xiang and Chunhong Pan;
    IEEE Geoscience and Remote Sensing Letters, 2020.

The full paper is available at: [https://doi.org/10.1109/LGRS.2020.3035844](https://doi.org/10.1109/LGRS.2020.3035844). 

## Highlights
- **General:** TDTS is a general method to exploit the spatial sparsity of aerial images for improving inference speed
- **Flop counter:** Enable to count the model's flops.

## Updates
   - 2020/11/18: first commit

## Installation
Our codes are mainly based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) 
and [spconv](https://github.com/traveller59/spconv), please refer to these repositories for installation and usage.

## Test
The inference command line on VisDrone-2019 val split:

    python tools/test_net.py \
        --config-file configs/visdrone_tdts/fcos_sparse_imprv_R_18_FPN_1x_800x1333_visdrone_cn_mw1.5-nms0.yaml \
        MODEL.WEIGHT fcos_sparse_imprv_R_18_FPN_1x_800x1333_visdrone_cn_mw1.5-nms0.pth \
        TEST.IMS_PER_BATCH 4    
        
## Train
To train a new model on VisDrone-2019 train split, run:

    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/visdrone_tdts/fcos_sparse_imprv_R_18_FPN_1x_800x1333_visdrone_cn_mw1.5-nms0.yaml \
        DATALOADER.NUM_WORKERS 4 \
        OUTPUT_DIR training_dir/fcos_sparse_imprv_R_18_FPN_1x_800x1333_visdrone_cn_mw1.5-nms0

## Models
Here we provide the following trained models.

Model | Backbone | Train Size| FPS@1 | FPS@8 | AP | AP50 | AP75 | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
TDTS_{dense}  | ResNet-18 | [640,800]x1333 | 30.3 | 25.1 | 23.00 | 41.98 | 22.93 | [download](xx)
TDTS_{sparse} | ResNet-18 | [640,800]x1333 | 40.0 | 51.1 | 22.68 | 41.22 | 22.70 | [download](xx)
TDTS_{dense}  | ResNet-50 | [512,640]x1024 | 33.2 | 34.0 | 21.91 | 40.25 | 21.79 | [download](xx)
TDTS_{sparse} | ResNet-50 | [512,640]x1024 | 35.8 | 47.9 | 21.71 | 39.81 | 21.63 |[download](xx)

* The FPS is tested with a RTX Titan GPU.

## Contributing to the project
Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{tdts,
  title   =  {Train in Dense and Test in Sparse: A Method for Sparse Object Detection in Aerial Images},
  author  =  {Ding, Kun and He, Guojin and Gu, Huxiang and Zhong, Zisha and Xiang, Shiming and Pan, Chuhong},
  booktitle =  {IEEE Geosci. Remote Sens. Lett.},
  year    =  {2020}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 
