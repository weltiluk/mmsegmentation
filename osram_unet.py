"""
python tools/train.py osram_unet.py
"""

"""Five-class UNet/FCN config for the prepared OSRAM dataset."""

_base_ = (
    './configs/unet/'
    'unet-s5-d16_fcn_4xb4-ce-1.0-dice-3.0-40k_hrf-256x256.py'
)

custom_imports = dict(imports=['projects.osram'], allow_failed_imports=False)

dataset_type = 'BaseSegDataset'
data_root = (
    '/workspaces/masterarbeit/masterarbeit/stratified_split/mixed_datasets/'
    'r4_s1_80_10_10_osram_png_norm_fix')

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

crop_size = (256, 256)
batch_size = 32
max_epochs = 200
val_interval = 1
checkpoint_interval = 10

# Preserve the requested UNet, FCN heads and CE 1.0 + Dice 3.0 loss. Only
# adapt both classifier outputs from two classes to five classes.
model = dict(
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(170, 170)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOsramAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
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
    sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
        _delete_=True,
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
        _delete_=True,
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
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=False,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=val_interval,
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=max_epochs,
        by_epoch=True,
    )
]

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True,
    ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=checkpoint_interval,
    ),
)

log_processor = dict(by_epoch=True)

work_dir = f'./results/osram_unet/{dataset_name}'