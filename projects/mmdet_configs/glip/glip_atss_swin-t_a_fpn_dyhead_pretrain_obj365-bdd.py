_base_ = [
    '../_base_/datasets/lvis_v0.5_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

lang_model_name = 'bert-base-uncased'

model = dict(
    type='GLIP',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        relu_before_extra_convs=True,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSVLFusionHead',
        lang_model_name=lang_model_name,
        num_classes=1203,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoderForGLIP',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
    ),
    language_model=dict(type='BertModel', name=lang_model_name,max_tokens=512),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            data_prefix=dict(img=''))))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        return_classes=True,
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/lvis_v1_val.json')
test_evaluator = val_evaluator
