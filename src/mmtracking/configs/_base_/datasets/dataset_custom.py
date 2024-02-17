dataset_type = 'CocoVideoDataset'
# classes = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus',
#            'car', 'cattle', 'dog', 'domestic cat', 'elephant', 'fox',
#            'giant panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
#            'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel',
#            'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra')
classes = ('bicycle', 'skateboard', 'e_scooter',)
data_root = '/content'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=False),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=16),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

 

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + '/annotations_train.json',
            img_prefix=data_root + '/train_yolo/',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=9,
                filter_key_img=False,
                method='uniform'),
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '/annotations_test.json',
        img_prefix=data_root + '/test_yolo/',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '/annotations_test.json',
        img_prefix=data_root + '/test_yolo/',
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(interval=1, metric='bbox')