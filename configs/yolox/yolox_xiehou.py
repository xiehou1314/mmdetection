_base_ = './yolox_s_8xb8-300e_coco.py'

# 需要修改的只有数据集，类别 batch-size
# img_scale = (640, 640)  # width, height

dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'D:/shared/tutor/Experimental/datasets/fujiazhuang/' # 数据的根路径。

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='/')))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='/')))

test_dataloader = val_dataloader