#include <cassert>
#include <iostream>
#include <torch/torch.h>
#include <math.h>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "include/structures/image_list.h"
#include "memory.h"


// MaskDINO2Encoder build_pixel_decoder(CfgNode cfg, ShapeSpec input_shape) {

// }


class MSDeformAttnTransformerEncoderOnly : torch::nn::Module {
public:
    MSDeformAttnTransformerEncoderOnly(int d = 256, int nhead = 8, int num_encoder_layers = 6,
                                       int dim_feedforward = 1024, float dropout = 0.1,
                                       std::string activation = "relu", int num_feature_levels = 4,
                                       int enc_n_points = 4) : d_model(d), nhead(nhead)
    {
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout,
                                                            activation, num_feature_levels,
                                                            nhead, enc_n_points);
        encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers);
        register_parameter("level_embed", torch::tensor({num_feature_levels, d_model}));
        reset_parameters();
    }

    void reset_parameters();

    double get_valid_ratio(torch::Tensor mask);

    torch::Tensor forward(torch::Tensor, torch::Tensor, int);

private:
    int d_model;
    int nhead;
    MSDeformAttnTransformerEncoderLayer encoder_layer;
    MSDeformAttnTransformerEncoder encoder;

};

void MSDeformAttnTransformerEncoderOnly::reset_parameters() {
    for (auto& p : parameters()) {
        if (p.dim() > 1)
            torch::nn::init::xavier_uniform_(p);
    }
    for (auto& m : modules()) {
        if (typeid(m).name() == "MSDeformAttn")
            static_cast<MSDeformAttn>(m).reset_parameters();
    torch::nn::init::normal_(this->named_parameters()["level_embed"]);
    }
}


struct MSDeformAttnFunction {
    // ====================================== LEFT OFF HERE: ===========================================
    // https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py
    
    torch::Tensor forward();
    std::vector<int> backward();

};


void ms_deform_attn_core_pytorch() {
    // ====================================== LEFT OFF HERE: ===========================================
    // https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py
}


class MSDeformAttn : torch::nn::Module {
public:
    MSDeformAttn(int, int, int, int);

    void reset_parameters();

    torch::Tensor forward(torch::Tensor, torch::Tensor, torch::Tensor,
                          torch::Tensor, torch::Tensor, std::initializer_list<torch::Tensor>);

private:
    int d_model;
    int n_levels;
    int n_heads;
    int n_points;
    int im2col_step = 128;
    torch::nn::Linear sampling_offsets;
    torch::nn::Linear attention_weights;
    torch::nn::Linear value_proj;
    torch::nn::Linear output_proj;

};

MSDeformAttn::MSDeformAttn(int d = 256, int n_levels = 4, int n_heads = 8, int n_points = 4) :
                           d_model(d), n_levels(n_levels), n_heads(n_heads), n_points(n_points)
{
    if (d_model % n_heads != 0)
        std::cerr << "d_model must be divisible by n_heads, but got " <<
                        d_model << " and " << n_heads;
    int _d_per_head = int(d_model / n_heads);
    if (_d_per_head & (_d_per_head - 1) == 0)
        std::cerr << "You'd better set d_model in MSDeformAttn to make the \
                        dimension of each attention head a power of 2 \
                        which is more efficient in our CUDA implementation.";

    torch::nn::Linear sampling_offsets = torch::nn::Linear(d_model, n_heads * n_levels * n_points * 2);
    torch::nn::Linear attention_weights = torch::nn::Linear(d_model, n_heads * n_levels * n_points);
    torch::nn::Linear value_proj = torch::nn::Linear(d_model, d_model);
    torch::nn::Linear output_proj = torch::nn::Linear(d_model, d_model);
    reset_parameters();
}

void MSDeformAttn::reset_parameters() {
    torch::nn::init::constant_(sampling_offsets->weight.data(), 0.);
    torch::Tensor thetas = torch::arange(n_heads, torch::kFloat32) * (2.0 * M_PI / n_heads);
    torch::Tensor grid_init = torch::stack({thetas.cos(), thetas.sin()}, -1);
    grid_init = (grid_init / get<0>(grid_init.abs().max(-1, true)))
                    .view({n_heads, 1, 1, 2})
                    .repeat({1, n_levels, n_points, 1});
    for (int i = 0; i < n_points; ++i)
        torch::mul(grid_init.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(),
                                    i,
                                    torch::indexing::Slice()}),
                                    i + 1);
    torch::NoGradGuard no_grad;
    sampling_offsets->bias = this->register_parameter("bias", grid_init.view({-1}));
    torch::nn::init::constant_(attention_weights->weight.data(), 0.);
    torch::nn::init::constant_(attention_weights->bias.data(), 0.);
    torch::nn::init::xavier_uniform_(value_proj->weight.data());
    torch::nn::init::constant_(value_proj->bias.data(), 0.);
    torch::nn::init::xavier_uniform_(output_proj->weight.data());
    torch::nn::init::constant_(output_proj->bias.data(), 0.);
}

torch::Tensor MSDeformAttn::forward(torch::Tensor query, torch::Tensor reference_pts, torch::Tensor input_flatten,
                                    torch::Tensor input_spatial_shapes, torch::Tensor input_level_start_index,
                                    std::initializer_list<torch::Tensor> input_padding_mask_list)
{
    int N = int(query.sizes()[0]);
    int Len_q = int(query.sizes()[1]);
    int Len_in = int(input_flatten.sizes()[1]);
    assert(Len_in == (input_spatial_shapes.index({torch::indexing::Slice(), 0}) * 
                      input_spatial_shapes.index({torch::indexing::Slice(), 1}))
                      .sum().item<int>());
    torch::Tensor value = value_proj(input_flatten);
    if (input_padding_mask_list.size() != 0) {
        const torch::Tensor& input_padding_mask = *input_padding_mask_list.begin();
        value = value.masked_fill(
            input_padding_mask.index({"...", torch::indexing::None}),
            float(0)
        );
    }

    value = value.view({N, Len_in, n_heads, int(d_model / n_heads)});
    torch::Tensor sampling_offsets_tensor = sampling_offsets(query).view({N, Len_q, n_heads, n_levels, n_points, 2});
    torch::Tensor attention_weights_tensor = attention_weights(query).view({N, Len_q, n_heads, n_levels * n_points});
    attention_weights_tensor = torch::nn::functional::softmax(attention_weights_tensor, -1).view({N, Len_q, n_heads, n_levels, n_points});

    torch::Tensor sampling_locations;
    if (reference_pts.sizes()[-1] == 2) {
        torch::Tensor offset_normalizer = torch::stack({input_spatial_shapes.index({"...", 1}),
                                                        input_spatial_shapes.index({"...", 0})}, -1);
        torch::Tensor *tmp1 = &reference_pts.index({torch::indexing::Slice(),
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice()});
        torch::Tensor *tmp2 = &offset_normalizer.index({torch::indexing::None,
                                                        torch::indexing::None,
                                                        torch::indexing::None,
                                                        torch::indexing::Slice(),
                                                        torch::indexing::None,
                                                        torch::indexing::Slice()});
        sampling_locations = *tmp1 + sampling_offsets_tensor / *tmp2;
    }
    else if (reference_pts.sizes()[-1] == 4) {
        torch::Tensor *tmp1 = &reference_pts.index({torch::indexing::Slice(),
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice(torch::indexing::None, 2)});
        torch::Tensor *tmp2 = &reference_pts.index({torch::indexing::Slice(),
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice(),
                                                   torch::indexing::None,
                                                   torch::indexing::Slice(2, torch::indexing::None)});
        sampling_locations = *tmp1 + torch::div(sampling_offsets_tensor, n_points) * 0.5 * *tmp2;
    }
    else {
        throw std::invalid_argument("Last dim of reference_points must be 2 or 4.");
    }
    torch::Tensor output;
    try {
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights_tensor, im2col_step
        );
    }
    catch (...) {
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights_tensor);
    }
    output = output_proj(output);
    return output;

    
}


class MSDeformAttnTransformerEncoderLayer : torch::nn::Module {

};


class MSDeformAttnTransformerEncoder : torch::nn::Module {

};


class MaskDINO2Encoder : torch::nn::Module {
public:
    MaskDINO2Encoder from_config(CfgNode cfg);
    torch::Tensor forward_features(torch::Tensor features, torch::Tensor masks);

};

