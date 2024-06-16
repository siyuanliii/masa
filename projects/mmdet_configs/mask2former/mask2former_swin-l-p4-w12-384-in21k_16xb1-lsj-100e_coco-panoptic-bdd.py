_base_ = ['./mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
backend_args = None
model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(num_queries=200, in_channels=[192, 384, 768, 1536]))

train_dataloader = dict(batch_size=1, num_workers=1)

# learning policy
max_iters = 737500
param_scheduler = dict(end=max_iters, milestones=[655556, 710184])

# Before 735001th iteration, we do evaluation every 5000 iterations.
# After 735000th iteration, we do evaluation every 737500 iterations,
# which means that we do evaluation at the end of training.'
interval = 5000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataset settings
data_root = 'data/bdd/'
dataset_type = 'CocoPanopticDataset'

# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/det_20/det_train_rm_anno_panoptic.json',
        ann_file='annotations/det_20/det_train_rm_anno_panoptic.json',
        data_prefix=dict(img='bdd100k/images/100k/train/', _delete_=True),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoPanopticMetric',
_delete_=True,
    format_only=True,
    ann_file=data_root + 'annotations/det_20/det_train_rm_anno_panoptic.json',
    outfile_prefix='./work_dirs/bdd_panoptic/test')