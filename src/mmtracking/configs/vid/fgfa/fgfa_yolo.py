_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/dataset_custom_fgfa.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='FGFA',
    detector=dict(
        type='YOLOX',
        init_cfg=dict(  # This line and the next is what you want to add
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        ),
        input_size=(640, 640),
        random_size_range=(15, 25),
        random_size_interval=10,

        # Backbone
        backbone=dict(
            type='CSPDarknet',
            deepen_factor=1.33,
            widen_factor=1.25
        ),

        # Neck
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4
        ),

        # BBox Head
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=3,  # Adjust this based on your dataset classes
            in_channels=320,
            feat_channels=320
        ),

        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)
        ),
        test_cfg=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.65)
        )
    ),

    motion=dict(
        type='FlowNetSimple',
        img_scale_factor=0.5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmtracking/pretrained_weights/flownet_simple.pth'
        )
    ),
    aggregator=dict(
        type='EmbedAggregator', num_convs=1, channels=320, kernel_size=3
    ),
    train_cfg=None,
    test_cfg=dict(key_frame_interval=10)
)

optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.00001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2)
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2, 5]
)
total_epochs = 13
evaluation = dict(metric=['bbox'], interval=1)
runner = {'type': 'IterBasedRunner', 'max_iters': 29476400}
checkpoint_config = dict(interval=250, by_epoch=False)
