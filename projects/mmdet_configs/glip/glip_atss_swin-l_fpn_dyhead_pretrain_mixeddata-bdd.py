_base_ = './glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'

model = dict(
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        drop_path_rate=0.4,
    ),
    neck=dict(in_channels=[384, 768, 1536]),
    bbox_head=dict(early_fuse=True, num_dyhead_blocks=8),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)

dataset_type = 'CocoDataset'
data_root = 'data/bdd/'

test_dataloader = dict(
    dataset=dict(
        test_mode=True,
        data_root=data_root,
        ann_file='annotations/det_20/det_train_rm_anno_panoptic.json',
        data_prefix=dict(img='bdd100k/images/100k/train/', _delete_=True),
    ))
test_evaluator = dict(
    format_only=True,
    ann_file=data_root + 'annotations/det_20/det_train_rm_anno_panoptic.json',
    outfile_prefix='/scratch/lisiyu/bdd_glip_rm_fp_05/test')