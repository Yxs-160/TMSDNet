# TMSDNet
Code of TMSDNet

## Create an environment ##
Directly run:
```
conda env create -f config/environment.yaml
conda activate tmsdnet
```

## Dataset ##
Download the [Rendered Images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) and [Voxelization (32)](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz) and decompress them into `$SHAPENET_IMAGE` and `$SHAPENET_VOXEL`

## Training ##
Here is a training code example
```
python train.py \
    --model image2voxel \
    --transformer_config config/tmsdnet-b.yaml \
    --annot_path data/ShapeNet.json \
    --model_path $SHAPENET_VOX \
    --image_path $SHAPENET_IMAGES \
    --gpus 1 \
    --precision 16 \
    --deterministic \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 4 \
    --check_val_every_n_epoch 1 \
    --accumulate_grad_batches 1 \
    --view_num 1 \
    --sample_batch_num 0 \
    --loss_type dice \
```
