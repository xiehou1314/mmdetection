_base_ = [
    './dino-4scale_r50_8xb2-12e_coco.py'
]

# python tools/train.py configs/dino/dino_xiehou.py
# python tools/train.py configs/dino/dino_xiehou.py --resume work_dirs/dino_xiehou/epoch_48.pth

METAINFO = {
        'classes':
        ('litter','default'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60),(220,20,60)]
    }
classes = ('default','litter')

num_classes=2
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'D:/shared/tutor/Experimental/datasets/fujiazhuang/' # 数据的根路径

max_epochs=80
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

backend_args=None

model = dict(
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)
