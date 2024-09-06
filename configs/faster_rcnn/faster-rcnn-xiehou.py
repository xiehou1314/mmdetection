# _base_ = './faster-rcnn_r50_fpn_8xb8-amp-lsj-200e_coco.py'
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# python tools/train.py configs/faster_rcnn/faster-rcnn-xiehou.py

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
data_root = 'D:/shared/tutor/Experimental/datasets/fujiazhuang/' # 数据的根路径。

backend_args=None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)))

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
    batch_size=1,
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
