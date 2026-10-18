"""DINOv3 ConvNeXt-Small + UPerNet, frozen backbone, OSRAM five classes.

Start through tools/osram_dinov3.py; see projects/osram_dinov3/README.md.
Data augmentation, losses and validation selection match the OSRAM UNet recipe.
"""

"""
train: (noch data_root hier anpassen je nach dataset)

python tools/train.py osram_dinov3_convnext.py \
  --amp \
  --work-dir results_dinov3/seed42/r9_s1_60_20_20_osram_final20_train60_longer_training \
&& \
python tools/osram_dinov3.py test \
  results_dinov3/seed42/r9_s1_60_20_20_osram_final20_train60_longer_training
"""

_base_ = './configs/_base_/default_runtime.py'
custom_imports = dict(
    imports=['projects.osram', 'projects.osram_dinov3'],
    allow_failed_imports=False)

dataset_type = 'BaseSegDataset'
# data_root = (
#     "/workspaces/masterarbeit/masterarbeit/stratified_split/real_dataset_splits_60_20_20"
# )
data_root = (
    "/workspaces/masterarbeit/masterarbeit/stratified_split/mixed_datasets/r9_s1_60_20_20_osram_final20_train60_longer_training"
)

dataset_name = data_root.rstrip('/').rsplit('/', maxsplit=1)[-1]

classes = ('background', 'class_1', 'class_2', 'class_3', 'class_4')
palette = [
    [0, 0, 0],
    [220, 20, 60],
    [0, 170, 255],
    [0, 180, 90],
    [255, 190, 0],
]
metainfo = dict(classes=classes, palette=palette)

crop_size = (512, 512)
batch_size = 8
# batch_size = 32
max_iters = 35000
val_interval = 150
checkpoint_interval = 2000
scheduler_end = max_iters

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOsramAnnotations'),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOsramAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='validation/images', seg_map_path='validation/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='ClasswiseIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
)
test_evaluator = val_evaluator
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=val_interval,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False,
    ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=checkpoint_interval,
        save_best='fg_mDice',
        rule='greater',
    ),
    visualization=dict(
        type='ClassBalancedSegVisualizationHook',
        class_ids=[1, 2, 3, 4],
    ),
)

custom_hooks = [
    dict(type='BestMetricsVisualizationHook'),
    dict(
        type='EMAEarlyStoppingHook',
        monitor='fg_mDice',
        rule='greater',
        min_delta=0.05, # 0.05%
        patience=20,    # validations
        momentum=0.8,   # 0.8*old_val + 0.2*new_val
        check_finite=True,
    ),
]

log_processor = dict(by_epoch=False)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='ThreePanelSegVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)


# RGB/ImageNet normalization, numerically equivalent to HF rescale + normalize.
# Padding is applied after normalization; no HF resizing or center crop.
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], bgr_to_rgb=True,
        size=crop_size, pad_val=0, seg_pad_val=255,
        test_cfg=dict(size_divisor=32)),
    backbone=dict(
        type='DINOv3ConvNeXtSmall',
        pretrained='./checkpoints/dinov3-convnext-small-pretrain-lvd1689m',
        frozen=True, local_files_only=True),
    decode_head=dict(
        type='UPerHead', in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3], channels=128, pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1, num_classes=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False,
                 loss_weight=1.0, loss_name='loss_ce'),
            dict(type='DiceLoss', loss_weight=3.0, loss_name='loss_dice')]),
    # Frozen DINOv3: train only the main decoder; no auxiliary head.
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(340, 340)))

# This is a decoder-training starting recipe, not yet tuned on OSRAM validation.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, begin=0, end=500, by_epoch=False),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, begin=500,
         end=scheduler_end, by_epoch=False)]
env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=42, deterministic=False)
load_from = None
resume = False
work_dir = './results_dinov3/seed42/real_dataset_splits_60_20_20'
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
