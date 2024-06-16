# Model Zoo


Here we provide the list of our current available models and their performance on various benchmarks.
For TAO TETA, we use public detections from [Detic-SwinB](https://github.com/open-mmlab/mmdetection/blob/main/projects/Detic_new/configs/detic_centernet2_swin-b_fpn_4x_lvis-base_in21k-lvis.py). For BDD MOT, we use public detections from [ByteTrack, Yolo-X](https://github.com/ifzhang/ByteTrack). For BDD MOTS, we use public detections from [UNINEXT-H](https://github.com/MasterBin-IIAU/UNINEXT).

### Zero-shot Association Test on TAO and BDD100K. 
|                                                        | TAO TETA |      | BDD MOT |       | BDD MOTS |       | model                                                                               |
|--------------------------------------------------------|:--------:|:----:|---------|-------|----------|-------|-------------------------------------------------------------------------------------|
|                                                        |  AssocA  | TETA | AssocA  | mIDF1 | AssocA   | mIDF1 |                                                                                     |
| [TETer](https://github.com/SysCV/tet/tree/main)        |   36.7   | 34.6 | 52.9    | 51.6  | -        | -     | -                                                                                   |
| [OVTrack](https://github.com/SysCV/ovtrack)            |   36.7   | 34.7 | -       | -     | -        | -     | -                                                                                   |
| [ByteTrack](https://github.com/ifzhang/ByteTrack)      |    -     |  -   | 51.5    | 54.8  | -        | -     | -                                                                                   |
| [UNINEXT](https://github.com/MasterBin-IIAU/UNINEXT)   |    -     |  -   | -       | 56.7  | 53.2     | 48.5  | -                                                                                   |
| MASA-R50                                               |   43.0   | 45.8 | 52.1    | 55.1  | 54.8     | 50.2  | [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/masa_r50.pth)        |
| MASA-Sam-B                                             |   44.3   | 46.5 | 53.1    | 55.6  | 54.1     | 49.7  | [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/sam_vitb_masa.pth)   |
| MASA-Sam-H                                             |   44.6   | 46.4 | 52.4    | 55.2  | 54.8     | 50.1  | [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/sam_vith_masa.pth)   |
| MASA-Detic                                             |   44.1   | 46.3 | 53.3    | 55.8  | 53.8     | 49.9  | [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/detic_masa.pth)      |
| MASA-GroundingDINO                                     |   44.6   | 46.7 | 53.1    | 55.7  | 53.3     | 48.9  | [HF 🤗](https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth)      |

* Note that during MASA training, we do not use any in-domain images. The results are slightly higher than we reported in the paper.

For more details, please refer to the [benchmark_test.md](benchmark_test.md) file.