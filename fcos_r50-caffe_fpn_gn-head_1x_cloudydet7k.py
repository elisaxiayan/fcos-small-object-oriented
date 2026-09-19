_base_ = './fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/CloudyDet7K/'
backend_args = None

metainfo = dict(classes=('ship', ))

# modify data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),  # 减小最小尺寸以适应弱小目标
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    iou_thrs=[0.5],  # 只计算IoU=0.5的召回率
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# modify model settings for small infrared targets
model = dict(
    backbone=dict(
        out_indices=(0, 1, 2, 3),  # 使用更早的特征层
    ),
    neck=dict(
        start_level=0,  # 从 P2 开始
        add_extra_convs='on_output',
        num_outs=5,
    ),
    bbox_head=dict(
        num_classes=1,  # 对应你的 'ship' 单类别
        strides=[4, 8, 16, 32, 64],  # 调整步长以适应小目标
        # 调整损失函数参数以适应弱小目标
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,  # 降低分数阈值以检测弱小目标
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

work_dir = './work_dirs/cloudydet_fcos/'