#include <iostream>
#include <torch/torch.h>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "image_list.h"
#include "memory.h"


// MaskDINO2Encoder build_pixel_decoder(CfgNode cfg, ShapeSpec input_shape) {

// }


class MSDeformAttnTransformerEncoderOnly : torch::nn::Module {

};


class MSDeformAttnTransformerEncoderLayer : torch::nn::Module {

};


class MSDeformAttnTransformerEncoder : torch::nn::Module {

};


class MaskDINO2Encoder : torch::nn::Module {
public:
    MaskDINO2Encoder from_config(CfgNode cfg);
    torch::Tensor forward_features(torch::Tensor features, torch::Tensor masks);

};

