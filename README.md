Industry deployable version of the findings from [Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation](https://arxiv.org/abs/2206.02777).

![alt text](https://github.com/ark2224/MaskDINO2CPP/images/franework.jpeg)

# H1 Model components
MaskDINO2 consists of three components: a backbone, a pixel decoder and a Transformer decoder.

- backbone: Define and register your backbone. You can follow the Swin Transformer as an example.

- pixel decoder: pixel decoder is actually the multi-scale encoder in DINO and Deformable DETR, which follows mask2former to call it pixel decoder. It is in MaskDINO2Encoder, you can change your multi-scale encoder. The returned values include:

    1. mask_features is the per-pixel embeddings with resolution 1/4 of the original image, obtained by fusing backbone 1/4 features and multi-scale encoder encoded 1/8 features. This is used to produce binary masks.
    2. multi_scale_features, which is the multi-scale inputs to the Transformer decoder. For ResNet-50 models with 4 scales, we use resolution 1/32, 1/16, and 1/8 but you can use arbitrary resolutions here, and follow DINO to additionally downsample 1/32 to get a 4th scale with 1/64 resolution. For 5-scale models with SwinL, we additional use 1/4 resolution features as in DINO.

- transformer decoder: it mainly follows DINO decoder to do detection and segmentation tasks. It is defined in maskdino/modeling/transformer_decoder.
