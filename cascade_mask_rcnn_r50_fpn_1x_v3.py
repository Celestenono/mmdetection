# model settings
model = dict(
    type='CascadeRCNN',  # The name of detector
    data_preprocessor=dict(  # The config of data preprocessor, usually includes image normalization and padding
        type='DetDataPreprocessor',  # The type of the data preprocessor, refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor
        mean=[123.675, 116.28, 103.53],  # Mean values used to pre-training the pre-trained backbone models, ordered in R, G, B
        std=[58.395, 57.12, 57.375],  # Standard variance used to pre-training the pre-trained backbone models, ordered in R, G, B
        bgr_to_rgb=True,  # whether to convert image from BGR to RGB
        pad_mask=True,  # whether to pad instance masks
        pad_size_divisor=32),  # The size of padded image should be divisible by ``pad_size_divisor``
    backbone=dict(  # The config of backbone
        type='ResNet',  # The type of backbone network. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.ResNet
        depth=50,  # The depth of backbone, usually it is 50 or 101 for ResNet and ResNext backbones.
        num_stages=4,  # Number of stages of the backbone.
        out_indices=(0, 1, 2, 3),  # The index of output feature maps produced in each stage
        frozen_stages=1,  # The weights in the first stage are frozen
        # norm_cfg=dict(  # The config of normalization layers.
        #     type='BN',  # Type of norm layer, usually it is BN or GN
        #     requires_grad=True),  # Whether to train the gamma and beta in BN
        # norm_eval=True,  # Whether to freeze the statistics in BN
        style='pytorch', # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 Conv, 'caffe' means stride 2 layers are in 1x1 Convs.
    	init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # The ImageNet pretrained backbone to be loaded
    neck=dict(
        type='FPN',  # The neck of detector is FPN. We also support 'NASFPN', 'PAFPN', etc. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.FPN for more details.
        in_channels=[256, 512, 1024, 2048],  # The input channels, this is consistent with the output channels of backbone
        out_channels=256,  # The output channels of each level of the pyramid feature map
        num_outs=5),  # The number of output scales
    rpn_head=dict(
        type='RPNHead',  # The type of RPN head is 'RPNHead', we also support 'GARPNHead', etc. Refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.dense_heads.RPNHead for more details.
        in_channels=256,  # The input channels of each input feature map, this is consistent with the output channels of neck
        feat_channels=256,  # Feature channels of convolutional layers in the head.
        anchor_generator=dict(  # The config of anchor generator
            type='AnchorGenerator',  # Most of methods use AnchorGenerator, SSD Detectors uses `SSDAnchorGenerator`. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py#L18 for more details
            scales=[8],  # Basic scale of the anchor, the area of the anchor in one position of a feature map will be scale * base_sizes
            ratios=[0.5, 1.0, 2.0],  # The ratio between height and width.
            strides=[4, 8, 16, 32, 64]),  # The strides of the anchor generator. This is consistent with the FPN feature strides. The strides will be taken as base_sizes if base_sizes is not set.
        bbox_coder=dict(  # Config of box coder to encode and decode the boxes during training and testing
            type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of the methods. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py#L13 for more details.
            target_means=[0.0, 0.0, 0.0, 0.0],  # The target means used to encode and decode boxes
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # The standard variance used to encode and decode boxes
        loss_cls=dict(  # Config of loss function for the classification branch
            type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/cross_entropy_loss.py#L201 for more details
            use_sigmoid=True,  # RPN usually performs two-class classification, so it usually uses the sigmoid function.
            loss_weight=1.0),  # Loss weight of the classification branch.
        loss_bbox=dict(  # Config of loss function for the regression branch.
            type='SmoothL1Loss',# Type of loss, we also support many IoU Losses and smooth L1-loss, etc. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/smooth_l1_loss.py#L56 for implementation.
            beta = 1.0 / 9.0,
            loss_weight=1.0)),  # Loss weight of the regression branch.
    roi_head= dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(  # RoI feature extractor for bbox regression.
            type='SingleRoIExtractor',  # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L13 for details.
            roi_layer=dict(  # Config of RoI Layer
                type='RoIAlign',  # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported. Refer to https://mmcv.readthedocs.io/en/latest/api.html#mmcv.ops.RoIAlign for details.
                output_size=7,  # The output size of feature maps.
                sampling_num=2),  # Sampling ratio when extracting the RoI features. 0 means adaptive ratio.
            out_channels=256,  # output channels of the extracted feature.
            featmap_strides=[4, 8, 16, 32]),  # Strides of multi-scale feature maps. It should be consistent with the architecture of the backbone.
        bbox_head=[
            dict(  # Config of box head in the RoIHead.
                type='Shared2FCBBoxHead',  # Type of the bbox head, Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L220 for implementation details.
                num_fcs=2,
                in_channels=256,  # Input channels for bbox head. This is consistent with the out_channels in roi_extractor
                fc_out_channels=1024,  # Output feature channels of FC layers.
                roi_feat_size=7,  # Size of RoI features
                num_classes=2,  # Number of classes for classification
                bbox_coder=dict(  # Box coder used in the second stage.
                    type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of the methods.
                    target_means=[0.0, 0.0, 0.0, 0.0],  # Means used to encode and decode box
                    target_stds=[0.1, 0.1, 0.2, 0.2]),  # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
                reg_class_agnostic=True,  # Whether the regression is class agnostic.
                loss_cls=dict(  # Config of loss function for the classification branch
                    type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc.
                    use_sigmoid=False,  # Whether to use sigmoid.
                    loss_weight=1.0),  # Loss weight of the classification branch.
                loss_bbox=dict(  # Config of loss function for the regression branch.
                    type='SmoothL1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
                    beta = 1.0,
                    loss_weight=1.0)),  # Loss weight of the regression branch.

            dict(  # Config of box head in the RoIHead.
                type='Shared2FCBBoxHead',  # Type of the bbox head, Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L220 for implementation details.
                num_fcs=2,
                in_channels=256,  # Input channels for bbox head. This is consistent with the out_channels in roi_extractor
                fc_out_channels=1024,  # Output feature channels of FC layers.
                roi_feat_size=7,  # Size of RoI features
                num_classes=2,  # Number of classes for classification
                bbox_coder=dict(  # Box coder used in the second stage.
                    type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of the methods.
                    target_means=[0.0, 0.0, 0.0, 0.0],  # Means used to encode and decode box
                    target_stds=[0.05, 0.05, 0.1, 0.1]),  # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
                reg_class_agnostic=True,  # Whether the regression is class agnostic.
                loss_cls=dict(  # Config of loss function for the classification branch
                    type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc.
                    use_sigmoid=False,  # Whether to use sigmoid.
                    loss_weight=1.0),  # Loss weight of the classification branch.
                loss_bbox=dict(  # Config of loss function for the regression branch.
                    type='SmoothL1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
                    beta = 1.0,
                    loss_weight=1.0)),  # Loss weight of the regression branch.
            dict(  # Config of box head in the RoIHead.
                type='Shared2FCBBoxHead',  # Type of the bbox head, Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L220 for implementation details.
                num_fcs=2,
                in_channels=256,  # Input channels for bbox head. This is consistent with the out_channels in roi_extractor
                fc_out_channels=1024,  # Output feature channels of FC layers.
                roi_feat_size=7,  # Size of RoI features
                num_classes=2,  # Number of classes for classification
                bbox_coder=dict(  # Box coder used in the second stage.
                    type='DeltaXYWHBBoxCoder',  # Type of box coder. 'DeltaXYWHBBoxCoder' is applied for most of the methods.
                    target_means=[0.0, 0.0, 0.0, 0.0],  # Means used to encode and decode box
                    target_stds=[0.033, 0.033, 0.067, 0.067]),  # Standard variance for encoding and decoding. It is smaller since the boxes are more accurate. [0.1, 0.1, 0.2, 0.2] is a conventional setting.
                reg_class_agnostic=True,  # Whether the regression is class agnostic.
                loss_cls=dict(  # Config of loss function for the classification branch
                    type='CrossEntropyLoss',  # Type of loss for classification branch, we also support FocalLoss etc.
                    use_sigmoid=False,  # Whether to use sigmoid.
                    loss_weight=1.0),  # Loss weight of the classification branch.
                loss_bbox=dict(  # Config of loss function for the regression branch.
                    type='SmoothL1Loss',  # Type of loss, we also support many IoU Losses and smooth L1-loss, etc.
                    beta = 1.0,
                    loss_weight=1.0)),  # Loss weight of the regression branch.
            ],
        mask_roi_extractor=dict(  # RoI feature extractor for mask generation.
            type='SingleRoIExtractor',  # Type of the RoI feature extractor, most of methods uses SingleRoIExtractor.
            roi_layer=dict(  # Config of RoI Layer that extracts features for instance segmentation
                type='RoIAlign',  # Type of RoI Layer, DeformRoIPoolingPack and ModulatedDeformRoIPoolingPack are also supported
                output_size=14,  # The output size of feature maps.
                sampling_num=2),  # Sampling ratio when extracting the RoI features.
            out_channels=256,  # Output channels of the extracted feature.
            featmap_strides=[4, 8, 16, 32]),  # Strides of multi-scale feature maps.
        mask_head=dict(  # Mask prediction head
            type='FCNMaskHead',  # Type of mask head, refer to https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.roi_heads.FCNMaskHead for implementation details.
            num_convs=2,  # Number of convolutional layers in mask head.
            in_channels=256,  # Input channels, should be consistent with the output channels of mask roi extractor.
            conv_out_channels=256,  # Output channels of the convolutional layer.
            num_classes=2,  # Number of class to be segmented.
            loss_mask=dict(  # Config of loss function for the mask branch.
                type='CrossEntropyLoss',  # Type of loss used for segmentation
                use_mask=True,  # Whether to only train the mask in the correct class.
                loss_weight=1.0))),  # Loss weight of mask branch.
    train_cfg = dict(  # Config of training hyperparameters for rpn and rcnn
        rpn=dict(  # Training config of rpn
            assigner=dict(  # Config of assigner
                type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for many common detectors. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14 for more details.
                pos_iou_thr=0.7,  # IoU >= threshold 0.7 will be taken as positive samples
                neg_iou_thr=0.3,  # IoU < threshold 0.3 will be taken as negative samples
                min_pos_iou=0.3,  # The minimal IoU threshold to take boxes as positive samples
                # match_low_quality=True,  # Whether to match the boxes under low quality (see API doc for more details).
                ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
            sampler=dict(  # Config of positive/negative sampler
                type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14 for implementation details.
                num=256,  # Number of samples
                pos_fraction=0.5,  # The ratio of positive samples in the total samples.
                neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                add_gt_as_proposals=False),  # Whether add GT as proposals after sampling.
            allowed_border=0,  # The border allowed after padding for valid anchors.
            pos_weight=-1,  # The weight of positive samples during training.
            debug=False),  # Whether to set the debug mode
        rpn_proposal=dict(  # The config to generate proposals during training
            nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
            nms_pre=2000,  # The number of boxes before NMS
            nms_post=2000,  # The number of boxes to be kept by NMS. Only work in `GARPNHead`.
            max_per_img=2000,  # The number of boxes to be kept after NMS.
            nms=dict( # Config of NMS
                type='nms',  # Type of NMS
                iou_threshold=0.7 # NMS threshold
                ),
            min_bbox_size=0),  # The allowed minimal box size
        rcnn=[
            dict(  # The config for the roi heads.
                assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
                    type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14 for more details.
                    pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
                    neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
                    min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
                    match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
                    ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
                sampler=dict(
                    type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14 for implementation details.
                    num=512,  # Number of samples
                    pos_fraction=0.25,  # The ratio of positive samples in the total samples.
                    neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                    add_gt_as_proposals=True
                ),  # Whether add GT as proposals after sampling.
                mask_size=28,  # Size of mask
                pos_weight=-1,  # The weight of positive samples during training.
                debug=False),  # Whether to set the debug mode
            dict(  # The config for the roi heads.
                assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
                    type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14 for more details.
                    pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
                    neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
                    min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
                    match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
                    ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
                sampler=dict(
                    type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14 for implementation details.
                    num=512,  # Number of samples
                    pos_fraction=0.25,  # The ratio of positive samples in the total samples.
                    neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                    add_gt_as_proposals=True
                ),  # Whether add GT as proposals after sampling.
                mask_size=28,  # Size of mask
                pos_weight=-1,  # The weight of positive samples during training.
                debug=False),  # Whether to set the debug mode
            dict(  # The config for the roi heads.
                assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
                    type='MaxIoUAssigner',  # Type of assigner, MaxIoUAssigner is used for all roi_heads for now. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14 for more details.
                    pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
                    neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
                    min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
                    match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
                    ignore_iof_thr=-1),  # IoF threshold for ignoring bboxes
                sampler=dict(
                    type='RandomSampler',  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14 for implementation details.
                    num=512,  # Number of samples
                    pos_fraction=0.25,  # The ratio of positive samples in the total samples.
                    neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                    add_gt_as_proposals=True
                ),  # Whether add GT as proposals after sampling.
                mask_size=28,  # Size of mask
                pos_weight=-1,  # The weight of positive samples during training.
                debug=False)  # Whether to set the debug mode
            ],
        stage_loss_weights=[1, 0.5, 0.25]),
    test_cfg = dict(  # Config for testing hyperparameters for rpn and rcnn
        rpn=dict(  # The config to generate proposals during testing
            nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
            nms_pre=1000,  # The number of boxes before NMS
            nms_post=1000,  # The number of boxes to be kept by NMS. Only work in `GARPNHead`.
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict( # Config of NMS
                type='nms',  #Type of NMS
                iou_threshold=0.7 # NMS threshold
                ),
            min_bbox_size=0),  # The allowed minimal box size
        rcnn=dict(  # The config for the roi heads.
            score_thr=0.05,  # Threshold to filter out boxes
            nms=dict(  # Config of NMS in the second stage
                type='nms',  # Type of NMS
                iou_thr=0.5),  # NMS threshold
            max_per_img=100,  # Max number of detections of each image
            mask_thr_binary=0.5),
        keep_all_stages=False))  # Threshold of mask prediction

# dataset settings
dataset_type = 'MyDataset'  # Dataset type, this will be used to define the dataset
data_root = "/scratch/nmoreau/glom_seg/data_for_training_patches/"  # Root path of data
backend_args = None # Arguments to instantiate the corresponding file backend

train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=True),  # Whether to use instance mask, True for instance segmentation
    dict(
        type='Resize',  # Pipeline that resizes the images and their annotations
        scale=(1024, 1024),  # The largest scale of the images
        keep_ratio=True  # Whether to keep the ratio between height and width
        ),
    dict(
        type='RandomFlip',  # Augmentation pipeline that flips the images and their annotations
        prob=0.5)  # The probability to flip
    # dict(type='PackDetInputs')  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
]
test_pipeline = [  # Testing data processing pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),  # Pipeline that resizes the images
    dict(
        type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(   # Train dataloader config
    batch_size=2,  # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # training data sampler
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.dataset.DefaultSampler.html#mmengine.dataset.DefaultSampler
        shuffle=True),  # randomly shuffle the training data in each epoch
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(  # Train dataset config
        type=dataset_type,
        data_root=data_root,
        ann_file="/train/annotations.json",  # Path of annotation file
        data_prefix=dict(img="/train/"),  # Prefix of image path
        filter_cfg=dict(filter_empty_gt=True, min_size=32),  # Config of filtering images and annotations
        pipeline=train_pipeline))
        # backend_args=backend_args))
val_dataloader = dict(  # Validation dataloader config
    batch_size=2,  # Batch size of a single GPU. If batch-size > 1, the extra padding area may influence the performance.
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    drop_last=False,  # Whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),  # not shuffle during validation and testing
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/val/annotations.json",
        data_prefix=dict(img="/val/"),
        test_mode=True,  # Turn on the test mode of the dataset to avoid filtering annotations or images
        pipeline=test_pipeline))
        # backend_args=backend_args))
val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + "/val/annotations.json",  # Annotation file path
    metric=['bbox', 'segm'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False)
    # backend_args=backend_args)
# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/test/annotations.json",
        data_prefix=dict(img="/test/"),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + "/test/annotations.json",
    metric=['bbox', 'segm'],  # Metrics to be evaluated
    format_only=True,  # Only format and save the results to coco json file
    outfile_prefix='/scratch/nmoreau/glom_seg/mmdetection_work_dirs/test')  # The prefix of output json files

train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=15,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type

optim_wrapper = dict(  # Optimizer wrapper config
    type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.005,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001),  # Weight decay of SGD
    clip_grad=dict(max_norm=35, norm_type=2),  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # Use linear policy to warmup learning rate
        start_factor=1.0 / 3, # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type='MultiStepLR',  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,   # Start from the first epoch
        end=15,  # End at the 12th epoch
        milestones=[8, 11])  # Epochs to decay the learning rate
        # gamma=0.1)  # The learning rate decay ratio
]

# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
#     logger=dict(type='LoggerHook', interval=50),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
#     param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
#     checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
#     sampler_seed=dict(type='DistSamplerSeedHook'),  # Ensure distributed Sampler shuffle is active
#     visualization=dict(type='DetVisualizationHook'))  # Detection Visualization Hook. Used to visualize validation and testing process prediction results

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50))
    # Optional: set moving average window size
log_processor = dict(
    type='LogProcessor', window_size=50)

default_scope = 'mmdet'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

env_cfg = dict(
    # cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(  # Multi-processing config
        mp_start_method='fork',  # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend='nccl'),  # Distribution configs
)

# vis_backends = [dict(type='LocalVisBackend')]  # Visualization backends. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# log_processor = dict(
#     type='LogProcessor',  # Log processor to process runtime logs
#     window_size=50,  # Smooth interval of log values
#     by_epoch=True)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.

log_level = 'INFO'  # The level of logging.
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.