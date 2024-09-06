_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# python tools/train.py configs/mask_rcnn/mask-rcnn-xiehou.py

# METAINFO = {
#         'classes':
#         ('litter','default'),
#         # palette is a list of color tuples, which is used for visualization.
#         'palette':
#         [(220, 20, 60),(220,20,60)]
#     }
# classes = ('default','litter')

METAINFO = {
        'classes':
        ('plastic_litter'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }
classes = ('plastic_litter')

num_classes=1
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
#data_root = 'D:/shared/tutor/Experimental/datasets/fujiazhuang/' # 数据的根路径
data_root = 'D:/shared/tutor/Experimental/datasets/Beach Plastic Litter Dataset version 1-256/'

val_interval=6
max_epochs=12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)

backend_args=None

model = dict(
    type='MaskRCNN',  # 检测器名
    roi_head=dict(  # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步
        bbox_head=dict(  # RoIHead 中 box head 的配置
            num_classes=2)))  # 分类的类别数量

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=8,
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
    # optimizer=dict(
    #     type='SGD',
    #     lr=0.0001,  # 0.0002 for DeformDETR
    #     weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[11],
#         gamma=0.1)
# ]