# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


test_dataset_tpye = 'BDDVideoDataset'

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

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'),
    dataset=dict(
        type=test_dataset_tpye,
        ann_file='data/bdd/annotations/box_track_20/box_track_val_cocofmt.json',
        data_prefix=dict(img_path='data/bdd/bdd100k/images/track/val/'),
        test_mode=True,
        pipeline=test_pipeline
    ))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='BDDTETAMetric',
    dataset_type=test_dataset_tpye,
    format_only=False,
    ann_file='data/bdd/annotations/box_track_20/box_track_val_cocofmt.json',
    scalabel_gt='data/bdd/annotations/scalabel_gt/box_track_20/val/',
    metric=['TETA'])
test_evaluator = val_evaluator


