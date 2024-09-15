# dataset settings
dataset_type = 'MASADataset'
data_root = 'data/sam/'
img_scale = (1024, 1024)

# data pipeline
train_pipeline = [
    dict(
        type='MixUniformRefFrameSample',
        num_ref_imgs=1,
        frame_range=0,
        filter_key_img=False),
    dict(
        type='MasaTransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='SeqRandomAffine',
            ),
            dict(
                type='SeqMixUp',
                img_scale=img_scale,
                ratio_range=(0.8, 1.6),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(type='YOLOXHSVRandomAug'),
            dict(
                type='RandomResize',
                scale=img_scale,
                ratio_range=(0.1, 2.0),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='RandomCrop', crop_size=img_scale, bbox_clip_border=False),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad',
                size=img_scale,
                pad_val=114,
                # If the image is three-channel, the pad value needs
                # to be set separately for each channel
            ),
            # dict(type='SeqCopyPaste'),
            dict(type='FilterMatchAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        ]),
    dict(type='PackMatchInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler'),  # image-based sampling
    dataset=dict(
        type='SeqMultiImageMixDataset',
        dataset=dict(
            type='RandomSampleConcatDataset',
            sampling_probs=[1],
            fixed_length=200000,
            datasets=[
                dict(
                    type=dataset_type,
                    ann_file='data/sam/sam_annotations/jsons/sa1b_coco_fmt_500k_bbox_anno.json',
                    data_prefix=dict(img='data/sam/batch0/'),
                    serialize_data=True,
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadMatchAnnotations'), ]
                )
            ]
    ),
        pipeline=train_pipeline

)
)

test_dataset_tpye = 'Taov1Dataset'

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=test_dataset_tpye,
        ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
        data_prefix=dict(img_path='data/tao/frames/'),
        test_mode=True,
        pipeline=test_pipeline
    ))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='TaoTETAMetric',
    dataset_type=test_dataset_tpye,
    format_only=False,
    ann_file='data/tao/annotations/tao_val_lvis_v1_classes.json',
    metric=['TETA'])
test_evaluator = val_evaluator


