_base_ = [
    '../dino/dino-4scale_r50_8xb2-12e_coco.py'
]

# python tools/train.py configs/kaggle/dino_kaggle_TrashCan.py
# python tools/train.py configs/kaggle/dino_kaggle_TrashCan.py --resume work_dirs/dino_xiehou/epoch_48.pth

METAINFO = {
        'classes':
        ('animal_crab','animal_eel','animal_etc','animal_fish','animal_shells','animal_starfish','plant',
           'rov','trash_bag','trash_bottle','trash_branch','trash_can','trash_clothing','trash_container',
           'trash_cup','trash_net','trash_pipe','trash_rope','trash_snack_wrapper','trash_tarp','trash_unknown_instance',
           'trash_wreckage'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255)]
    }
classes = ('animal_crab','animal_eel','animal_etc','animal_fish','animal_shells','animal_starfish','plant',
           'rov','trash_bag','trash_bottle','trash_branch','trash_can','trash_clothing','trash_container',
           'trash_cup','trash_net','trash_pipe','trash_rope','trash_snack_wrapper','trash_tarp','trash_unknown_instance',
           'trash_wreckage')

num_classes=22
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
# 本地训练
#data_root='../../datasets/TrashCan/dataset/dataset/instance_version/'
# kaggle训练
data_root='../TrashCan/instance_version/'

max_epochs=80
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

backend_args=None

model = dict(
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_train_trashcan.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='instances_val_trashcan.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val_trashcan.json',
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
