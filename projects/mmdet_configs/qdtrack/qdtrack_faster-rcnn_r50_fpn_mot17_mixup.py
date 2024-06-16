_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_4e_base.py',
    # '../_base_/datasets/mot_challenge.py',
]
# dataset settings
dataset_type = 'CocoInsDataset'
data_root = 'data/MOT17/'
img_scale = (1088, 1088)

backend_args = None
# data pipeline
train_pipeline = [
    dict(
        type='MixUniformRefFrameSample',
        num_ref_imgs=1,
        frame_range=0,
        filter_key_img=False),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            # dict(
            #     type='SeqMosaic',
            #     img_scale=img_scale,
            #     pad_val=114.0,
            #     bbox_clip_border=False),
            dict(
                type='SeqRandomAffine',
                # scaling_ratio_range=(0.1, 2),
                # border=(-img_scale[0] // 2, -img_scale[1] // 2),
                # bbox_clip_border=False
            ),
            dict(
                type='SeqMixUp',
                img_scale=img_scale,
                ratio_range=(0.8, 1.6),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(
                type='RandomResize',
                scale=img_scale,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='PhotoMetricDistortion'),
            dict(type='RandomCrop', crop_size=img_scale, bbox_clip_border=False),
            dict(type='RandomFlip', prob=0.5),
        ]),

    dict(type='PackMatchInputs')
]
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackMatchInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
  # image-based sampling
    dataset=dict(
        type='SeqMultiImageMixDataset',
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # visibility_thr=-1,
        ann_file='annotations/half-train_cocoformat.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        data_prefix=dict(img='train'),
        metainfo=dict(classes=('pedestrian', )),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadMatchAnnotations'),]),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type='MOTChallengeDataset',
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# evaluator

# evaluator
val_evaluator = dict(type='CocoVideoMetric', metric=['bbox'], classwise=True)
    # dict(type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'])


test_evaluator = val_evaluator
# The fluctuation of HOTA is about +-1.
randomness = dict(seed=6)
