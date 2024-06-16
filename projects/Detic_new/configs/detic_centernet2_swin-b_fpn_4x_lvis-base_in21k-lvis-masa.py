_base_ = './detic_centernet2_r50_fpn_4x_lvis_in21k-lvis.py'

keys_to_delete = [key for key in _base_.keys() if key != 'model']
for key in keys_to_delete:
    del _base_[key]

image_size_det = (896, 896)
image_size_cls = (448, 448)

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False),
    neck=dict(in_channels=[256, 512, 1024]),
    test_cfg=dict(
        rpn=dict(
            score_thr=0.0001,
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms',
                     iou_threshold=0.5,
                     class_agnostic=True,
                     split_thr=100000),
            max_per_img=100,
            mask_thr_binary=0.5)

    )
)
