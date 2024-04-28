_base_ = [
    '../_base_/models/pseudoclip.py', '../_base_/datasets/ade20k_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

img_size = 512
in_channels = 512
out_indices = [11]

base_class = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 
21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 
43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 
66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 
88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 
108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 
126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 
143, 144, 146, 147, 148, 149]
novel_class = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 
99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 
115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 
131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
147, 148, 149]

num_classes = len(base_class)

pretrained = '/pretrained/ViT-B-16.pt'

model = dict(
    type='ZegCLIP',
    pretrained=pretrained, 
    pretrained_text=pretrained, 
    context_length=77,
    # text_dim=512,
    # score_concat_index=2,
    backbone=dict(
        type='LoRACLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=img_size,
        out_indices=out_indices,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextEncoder',
        context_length=77,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    decode_head=dict(
        type='ATMSingleHeadSeg', #ATMSingleHeadSeg
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=3,
        num_heads=8,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=512,
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3, 
            mask_weight=100.0, #20.0
            dice_weight=1.0,
            loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)), 
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    ft_backbone = False,
    exclude_key='lora',
    load_text_embedding='configs/_base_/datasets/text_embedding/ade20k_150_multi.npy',
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00001, weight_decay=0.01, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        }))

data = dict(samples_per_gpu=6,
            workers_per_gpu=6,)

