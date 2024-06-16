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
)

dataset_type = 'CocoDataset'
data_root = 'data/uvo/'

test_dataloader = dict(
    dataset=dict(
        test_mode=True,
        data_root=data_root,
        ann_file='annotations/UVO_frame_val_coco_pano_cats.json',
        data_prefix=dict(img='uvo_det/uvo_videos_sparse_frames/', _delete_=True),
    ))
test_evaluator = dict(
    format_only=True,
    ann_file='data/uvo/annotations/UVO_frame_val_coco_pano_cats.json',
    outfile_prefix='/scratch/lisiyu/uvo_glip/test')