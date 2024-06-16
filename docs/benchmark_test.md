
# Testing on Standard Benchmarks

This document describes how to run MASA for the standard benchmarks testing.


## Prepare Datasets



It is recommended to symlink the dataset root to `$MASA/data`, and the model root to `$MASA/saved_models`.
If your folder structure is different, you may need to change the corresponding paths in config files.

### Download TAO
a. Please follow [TAO download](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) instructions.

b. Please download converted TAO annotations and put them in `data/tao/annotations/`.

You can download the annotations `tao_val_lvis_v05_classes.json` from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/tao_val_lvis_v05_classes.json). 

You can download the annotations `tao_val_lvis_v1_classes.json` from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/tao_val_lvis_v1_classes.json). 

Note that the original TAO annotations has some mistakes regarding class names. We have fixed the class names in the converted annotations and make it consistent with the LVIS dataset.
##### Optional: generate the annotations by yourself
You can also generate the annotations by yourself. Please refer to the instructions [here](https://github.com/SysCV/ovtrack/blob/main/docs/GET_STARTED.md).

#### Symlink the data
Our folder structure follows

```
├── masa
├── tools
├── configs
├── data
    ├── tao
        ├── frames
            ├── train
            ├── val
            ├── test
        ├── annotations

|── saved_models # saved_models are the folder to save downloaded pretrained models and also the models you trained.
    ├── pretrain_weights
    ├── masa_models
```

It will be easier if you create the same folder structure.


### Download BDD100K
We present an example based on [BDD100K](https://www.vis.xyz/bdd100k/) dataset. Please first download the images and annotations from the [official website](https://doc.bdd100k.com/download.html). 

On the download page, the required data and annotations are

- `mot` set images: `MOT 2020 Images`
- `mot` set annotations: `MOT 2020 Labels`
- `mots` set images: `MOTS 2020 Images`
- `mots` set annotations: `MOTS 2020 Labels`

#### Symlink the data

It is recommended to symlink the dataset root to `$MASA/data`.
The official BDD100K annotations are in the format of [scalabel](https://doc.bdd100k.com/format.html). Please put the scalabel annotations file udner the `scalabel_gt` folder.
Our folder structure follows

```
├── masa
├── tools
├── configs
├── data
│   ├── bdd
│   │   ├── bdd100k
            ├── images  
                ├── track 
                    |── val
        ├── annotations 
        │   ├── box_track_20
        │   ├── det_20
        │   ├── scalabel_gt
                |── box_track_20
                    |── val
                |── seg_track_20
                    |── val
```

#### Convert annotations to COCO style

The official BDD100K annotations are in the format of [scalabel](https://doc.bdd100k.com/format.html).

You can directly download the converted annotations: [mot](https://huggingface.co/dereksiyuanli/masa/resolve/main/bdd_box_track_val_cocofmt.json) and [mots](https://huggingface.co/dereksiyuanli/masa/resolve/main/bdd_seg_track_val_cocofmt.json) and put them in the `data/bdd/annotations/` folder.


(Optional) If you want to convert the annotations by yourself, you can use bdd100k toolkit. Please install the bdd100k toolkit by following the instructions [here](https://github.com/bdd100k/bdd100k).
Then, you can run the following commands to convert the annotations to COCO format.
```bash
mkdir data/bdd/annotations/box_track_20
python -m bdd100k.label.to_coco -m box_track -i data/bdd/annotations/scalabel_gt/box_track_20/${SET_NAME} -o data/bdd/annotations/box_track_20/bdd_box_track_${SET_NAME}_cocofmt.json
```
The `${SET_NAME}` here can be one of ['train', 'val', 'test'].

### Download the public detections
Create a folder named 'results' under the root.
```bash
mkdir results
```
Download the public detections from [here](https://huggingface.co/dereksiyuanli/masa/resolve/main/public_dets_masa.zip) and unzip it under the 'results' folder.

## Run MASA
This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

### Test a model with COCO-format

Note that, in this repo, the evaluation metrics are computed with COCO-format.
But to report the results on BDD100K, evaluating with BDD100K-format is required.

- single GPU
- single node multiple GPU

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--cfg-options]

# multi-gpu testing
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--cfg-options]
```

Optional arguments:
- `--cfg-options`: If specified, some setting in the used config will be overridden.


#### Test on TAO TETA benchmark

We provide the config file for testing on the TAO TETA benchmark. Use MASA-GroundingDINO for example.

```angular2html
tools/dist_test.sh configs/masa-gdino/tao_teta_test/masa_gdino_swinb_tao_test_detic_dets.py saved_models/masa_models/gdino_masa.pth 8 
```

#### Test on Open-vocabulary MOT benchmark
We provide the config file for testing on the open-vocabulary MOT benchmark. Use MASA-Detic for example.
```angular2html
tools/dist_test.sh configs/masa-detic/open_vocabulary_mot_test/masa_detic_swinb_open_vocabulary_test.py saved_models/masa_models/detic_masa.pth 8 
```

#### Test on BDD100K MOT benchmark
We provide the config file for testing on the BDD100K MOT benchmark, with ByteTrack's YoloX prediction. Use MASA-GroundingDINO for example.

```angular2html
tools/dist_test.sh configs/masa-gdino/bdd_test/masa_gdino_bdd_mot_test.py saved_models/masa_models/gdino_masa.pth 8 
```

#### Test on BDD100K MOTS benchmark
We provide the config file for testing on the BDD100K MOTS benchmark, with UNINEXT's prediction. Use MASA-GroundingDINO for example.

```angular2html
tools/dist_test.sh configs/masa-gdino/bdd_test/masa_gdino_bdd_mots_test.py saved_models/masa_models/gdino_masa.pth 8
```

For other models, you can replace the `${CONFIG_FILE}` and `${CHECKPOINT_FILE}` with the corresponding paths.
