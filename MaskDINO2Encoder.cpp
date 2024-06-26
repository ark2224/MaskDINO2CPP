﻿#include <array>
#include <cassert>
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <vector>
#include "backbone.h"
#include "detectron2.h"
#include "utils/modules.h"
#include "utils/position_encoding.h"
#include "matcher.h"
#include "memory.h"


class MSDeformAttnTransformerEncoderImpl : torch::nn::Module {
/*
    MSDeformAttn Transformer encoder in deformable detr
    This is the first half of the transformer
    Supporting functions and classes included in this file, below
*/
public:
    MSDeformAttnTransformerEncoderImpl() {  }
    MSDeformAttnTransformerEncoderImpl(
        torch::nn::AnyModule& encoder_layer,
        int const& num_l) :
        num_layers(num_l),
        torch::nn::Module()
        {
            layers = get_clones(encoder_layer, num_layers);
        }

    // class methods:
    static torch::Tensor get_reference_points(std::vector< std::pair<int, int> >&,
                                              torch::Tensor&,
                                              torch::Device);

    torch::Tensor forward(torch::Tensor&,
                          std::vector< std::pair<int, int> >&,
                          torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&);

private:
    torch::nn::Sequential
        layers = nullptr;
    int
        num_layers;
};
TORCH_MODULE(MSDeformAttnTransformerEncoder);

torch::Tensor MSDeformAttnTransformerEncoderImpl::get_reference_points(
    std::vector< std::pair<int, int> >& spatial_shapes,
    torch::Tensor& valid_ratios,
    torch::Device device)
{
    std::vector<torch::Tensor> reference_points_vector;
    int lvl = 0;
    for (auto const& shape : spatial_shapes) {
        int H_ = std::get<0>(shape);
        int W_ = std::get<1>(shape);

        std::vector<torch::Tensor> ref_xy = torch::meshgrid({
            torch::linspace(0.5, H_ - 0.5, H_, 
                            torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(device)),
            torch::linspace(0.5, W_ - 0.5, W_,
                            torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(device))
        });

        torch::Tensor ref_y = ref_xy[0];
        torch::Tensor ref_x = ref_xy[1];
        ref_y = ref_y.reshape({-1}).index({torch::indexing::None})
                / (H_ * valid_ratios.index({torch::indexing::Slice(),
                                            torch::indexing::None,
                                            lvl,
                                            1}));
        ref_x = ref_x.reshape({-1}).index({torch::indexing::None})
                / (W_ * valid_ratios.index({torch::indexing::Slice(),
                                            torch::indexing::None,
                                            lvl,
                                            0}));
        torch::Tensor ref = torch::stack({ref_x, ref_y}, -1);
        reference_points_vector.push_back(ref);
        ++lvl;
    }
    torch::Tensor reference_points = torch::cat(reference_points_vector, 1);
    reference_points = reference_points.index({torch::indexing::Slice(),
                                               torch::indexing::Slice(),
                                               torch::indexing::None})
                       * valid_ratios.index({torch::indexing::Slice(),
                                             torch::indexing::None});
    return reference_points;
}

torch::Tensor MSDeformAttnTransformerEncoderImpl::forward(
    torch::Tensor& src,
    std::vector<std::pair<int, int>>& spatial_shapes,
    torch::Tensor& level_start_index,
    torch::Tensor& valid_ratios,
    torch::Tensor& pos,
    torch::Tensor& padding_mask
)
{
    torch::Tensor output = src;
    torch::Tensor reference_points = get_reference_points(spatial_shapes,
                                                          valid_ratios,
                                                          src.device());
    output = layers->forward(output,
                             pos,
                             reference_points,
                             spatial_shapes,
                             level_start_index,
                             padding_mask);
    return output;
}


class MSDeformAttnFunction : torch::autograd::Function<torch::Tensor> {
public:
    // class methods:
    static torch::Tensor forward(torch::Tensor&,
                                 torch::Tensor&,
                                 torch::Tensor&,
                                 torch::Tensor&,
                                 torch::Tensor&,
                                 int&);
    static std::vector<int> backward(torch::Tensor&);
};

torch::Tensor MSDeformAttnFunction::forward(
    torch::Tensor& value,
    torch::Tensor& input_spatial_shapes,
    torch::Tensor& input_level_start_index,
    torch::Tensor& sampling_locations,
    torch::Tensor& attention_weights_tensor,
    int& im2col_step
)
{
    // Optional: download pythonic version of ms_deform_attn_cuda_forward and save_for_backward from:
    // https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/cuda/ms_deform_attn_cuda.h
    torch::Tensor output = ms_deform_attn_cuda_forward(
        value,
        input_spatial_shapes,
        input_level_start_index,
        sampling_locations,
        attention_weights_tensor,
        im2col_step
    );
    save_for_backward(value,
                      input_spatial_shapes,
                      input_level_start_index,
                      sampling_locations,
                      attention_weights_tensor);
    return output;
}

std::vector<int> MSDeformAttnFunction::backward(torch::Tensor& grad_output) {
    torch::Tensor value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights_tensor = this->saved_tensors();
    int grad_value, grad_sampling_loc, grad_attn_weight = 
        ms_deform_attn_cuda_backward(value,
                                     input_spatial_shapes,
                                     input_level_start_index,
                                     sampling_locations,
                                     attention_weights_tensor,
                                     grad_ouput,
                                     im2col_step);
    return std::vector<int> {grad_value, grad_sampling_loc, grad_attn_weight};
}


torch::Tensor ms_deform_attn_core_pytorch(torch::Tensor& value,
                                          torch::Tensor& value_spatial_shapes,
                                          torch::Tensor& sampling_locations,
                                          torch::Tensor& attention_weights)
{
    /*
        Optional: feel free to refer to link below for reference to deformable attention using CUDA gpu
    */
    // https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py
    int N_ = int(value.sizes()[0]);
    int S_ = int(value.sizes()[1]);
    int M_ = int(value.sizes()[2]);
    int D_ = int(value.sizes()[3]);

    int Lq_ = int(sampling_locations.sizes()[1]);
    int L_ = int(sampling_locations.sizes()[3]);
    int P_ = int(sampling_locations.sizes()[4]);

}


class MSDeformAttnImpl : torch::nn::Module {
/*
    Multi-Scale Deformable Attention Module: provides memory within transformer in series of inferences
    :param d_model      hidden dimension
    :param n_levels     number of feature levels
    :param n_heads      number of attention heads
    :param n_points     number of sampling points per attention head per feature level
*/
public:
    MSDeformAttnImpl() {  } //default constructor for other classes' initialization
    MSDeformAttnImpl(int const&,
                     int const&,
                     int const&,
                     int const&);
    // class methods:
    void reset_parameters();
    torch::Tensor forward(torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          std::initializer_list<torch::Tensor>&);
private:
    int
        d_model,
        n_levels,
        n_heads,
        n_points,
        im2col_step = 128;
    torch::nn::Linear
        sampling_offsets = nullptr,
        attention_weights = nullptr,
        value_proj = nullptr,
        output_proj = nullptr;
};
TORCH_MODULE(MSDeformAttn);

MSDeformAttnImpl::MSDeformAttnImpl(
    int const& d = 256,
    int const& n_levels = 4,
    int const& n_heads = 8,
    int const& n_points = 4
) : d_model(d),
    n_levels(n_levels),
    n_heads(n_heads),
    n_points(n_points)
{
    /*
        Main constructor for MSDeformAttn
    */
    if (d_model % n_heads != 0)
        std::cerr << "d_model must be divisible by n_heads, but got " <<
                    d_model << " and " << n_heads;
    int _d_per_head = int(d_model / n_heads);
    if (_d_per_head && (_d_per_head - 1) == 0)
        std::cerr << "You'd better set d_model in MSDeformAttn to make the \
                     dimension of each attention head a power of 2 \
                     which is more efficient in our CUDA implementation.";

    torch::nn::Linear sampling_offsets = torch::nn::Linear(d_model,
                                                           n_heads*n_levels*n_points*2);
    torch::nn::Linear attention_weights = torch::nn::Linear(d_model,
                                                            n_heads*n_levels*n_points);
    torch::nn::Linear value_proj = torch::nn::Linear(d_model, d_model);
    torch::nn::Linear output_proj = torch::nn::Linear(d_model, d_model);
    reset_parameters();
}

void MSDeformAttnImpl::reset_parameters() {
    torch::nn::init::constant_(sampling_offsets->weight.data(), 0.);
    torch::Tensor thetas = torch::arange(n_heads, torch::kFloat32)
                           * (2.0 * M_PI / n_heads);
    torch::Tensor grid_init = torch::stack({thetas.cos(), thetas.sin()}, -1);
    grid_init = (grid_init / std::get<0>(grid_init.abs().max(-1, true)))
                .view({n_heads, 1, 1, 2})
                .repeat({1, n_levels, n_points, 1});
    for (int i = 0; i < n_points; ++i) {
        torch::mul(grid_init.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(),
                                    i,
                                    torch::indexing::Slice()}),
                                   i + 1);
    }
    torch::NoGradGuard no_grad;
    sampling_offsets->bias = this->register_parameter("bias", grid_init.view({-1}));
    torch::nn::init::constant_(attention_weights->weight.data(), 0.);
    torch::nn::init::constant_(attention_weights->bias.data(), 0.);
    torch::nn::init::xavier_uniform_(value_proj->weight.data());
    torch::nn::init::constant_(value_proj->bias.data(), 0.);
    torch::nn::init::xavier_uniform_(output_proj->weight.data());
    torch::nn::init::constant_(output_proj->bias.data(), 0.);
}

torch::Tensor MSDeformAttnImpl::forward(
    torch::Tensor& query,
    torch::Tensor& reference_pts,
    torch::Tensor& input_flatten,
    torch::Tensor& input_spatial_shapes,
    torch::Tensor& input_level_start_index,
    std::initializer_list<torch::Tensor>& input_padding_mask_list)
{
/*
Parameters descriptions:
    :param query                       (N, Length_{query}, C)
    :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                    or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
    :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
    :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
    :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
    :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

    :return output                     (N, Length_{query}, C)
*/
    int N = int(query.sizes()[0]);
    int Len_q = int(query.sizes()[1]);
    int Len_in = int(input_flatten.sizes()[1]);

    assert(Len_in == 
        (
            input_spatial_shapes.index({torch::indexing::Slice(), 0})
            * input_spatial_shapes.index({torch::indexing::Slice(), 1})
        ).sum().item<int>());
    
    torch::Tensor value = value_proj(input_flatten);

    if (input_padding_mask_list.size() != 0) {
        const torch::Tensor& input_padding_mask = *input_padding_mask_list.begin();
        value = value.masked_fill(
            input_padding_mask.index({"...", torch::indexing::None}),
            float(0)
        );
    }

    value = value.view({N, Len_in, n_heads, int(d_model / n_heads)});

    torch::Tensor sampling_offsets_tensor =
        sampling_offsets(query).view({N, Len_q, n_heads, n_levels, n_points, 2});
    torch::Tensor attention_weights_tensor =
        attention_weights(query).view({N, Len_q, n_heads, n_levels * n_points});
    attention_weights_tensor = 
        torch::nn::functional::softmax(attention_weights_tensor, -1)
        .view({N, Len_q, n_heads, n_levels, n_points});

    torch::Tensor sampling_locations;
    if (reference_pts.sizes()[-1] == 2) {

        torch::Tensor offset_normalizer = 
            torch::stack({input_spatial_shapes.index({"...", 1}),
                          input_spatial_shapes.index({"...", 0})}, -1);
        torch::Tensor *tmp1 =
            &reference_pts.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::None,
                                  torch::indexing::Slice(),
                                  torch::indexing::None,
                                  torch::indexing::Slice()});
        torch::Tensor *tmp2 =
            &offset_normalizer.index({torch::indexing::None,
                                      torch::indexing::None,
                                      torch::indexing::None,
                                      torch::indexing::Slice(),
                                      torch::indexing::None,
                                      torch::indexing::Slice()});
        sampling_locations = *tmp1 + sampling_offsets_tensor / *tmp2;
    }
    else if (reference_pts.sizes()[-1] == 4) {

        torch::Tensor *tmp1 =
            &reference_pts.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::None,
                                  torch::indexing::Slice(),
                                  torch::indexing::None,
                                  torch::indexing::Slice(torch::indexing::None, 2)});
        torch::Tensor *tmp2 =
            &reference_pts.index({torch::indexing::Slice(),
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
        MSDeformAttnFunction tmp = MSDeformAttnFunction();
        output = tmp.forward(value,
                             input_spatial_shapes,
                             input_level_start_index,
                             sampling_locations,
                             attention_weights_tensor,
                             im2col_step);
    }
    catch (...) {
        output = ms_deform_attn_core_pytorch(value,
                                             input_spatial_shapes,
                                             sampling_locations,
                                             attention_weights_tensor);
    }
    output = output_proj(output);
    return output;
}


class MSDeformAttnTransformerEncoderLayer : torch::nn::Module {
/*
    Sub-class for individual layer of the MaskDINO's Transformer's Decoder
*/
public:
    MSDeformAttnTransformerEncoderLayer(int const&,
                                        int const&,
                                        float const&,
                                        std::string const&,
                                        int const&,
                                        int const&,
                                        int const&);
    // class methods:
    static torch::Tensor with_pos_embed(torch::Tensor& tensor,
                                        torch::Tensor& pos)
    { 
        if (pos.defined())
            return tensor + pos;
        else
            return tensor;
    }

    torch::Tensor forward_ffn(torch::Tensor& src)
    {
        torch::Tensor src2 = linear2(dropout2(activation_layer->forward(linear1(src))));
        src = src + dropout3(src2);
        src = norm2(src);
        return src;
    }

    torch::Tensor forward(torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          torch::Tensor&,
                          std::vector<torch::Tensor>&,
                          torch::Tensor&,
                          std::initializer_list<torch::Tensor>&);

private:
    MSDeformAttn self_attn;
    torch::nn::Dropout
        dropout1 = nullptr,
        dropout2 = nullptr,
        dropout3 = nullptr;
    torch::nn::LayerNorm
        norm1 = nullptr,
        norm2 = nullptr;
    torch::nn::Linear
        linear1 = nullptr,
        linear2 = nullptr;
    torch::nn::Sequential
        activation_layer = nullptr;
};

MSDeformAttnTransformerEncoderLayer::MSDeformAttnTransformerEncoderLayer(
    int const& d_model = 256,
    int const& d_ffn = 1024,
    float const& dropout = 0.1,
    std::string const& activation = "relu",
    int const& n_levels = 4,
    int const& n_heads = 8,
    int const& n_points = 4)
: torch::nn::Module()
{
    /*
        Main class constructor for the Encoder Layer
    */
    self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points);
    dropout1 = torch::nn::Dropout(dropout);
    norm1 = torch::nn::LayerNorm(d_model);
    linear1 = torch::nn::Linear(d_model, d_ffn);
    if (!activation.compare("relu"))
        activation_layer = torch::nn::Sequential(torch::nn::ReLU());
    else if (!activation.compare("gelu"))
        activation_layer = torch::nn::Sequential(torch::nn::GELU());
    else if (!activation.compare("glu"))
        activation_layer = torch::nn::Sequential(torch::nn::GLU());
    else if (!activation.compare("prelu"))
        activation_layer = torch::nn::Sequential(torch::nn::PReLU());
    else if (!activation.compare("selu"))
        activation_layer = torch::nn::Sequential(torch::nn::SELU());
    dropout2 = torch::nn::Dropout(dropout);
    linear2 = torch::nn::Linear(d_ffn, d_model);
    dropout3 = torch::nn::Dropout(dropout);
    norm2 = torch::nn::LayerNorm(d_model);
}

torch::Tensor MSDeformAttnTransformerEncoderLayer::forward(
    torch::Tensor& src,
    torch::Tensor& pos,
    torch::Tensor& reference_pts,
    torch::Tensor& input_flatten,
    std::vector<torch::Tensor>& input_spatial_shapes,
    torch::Tensor& input_level_start_index,
    std::initializer_list<torch::Tensor>& input_padding_mask_list)
{
    torch::Tensor Spatial_shapes = 
        torch::from_blob(input_spatial_shapes.data(),
                         input_spatial_shapes.size(),
                         torch::TensorOptions().dtype(torch::kLong).device(src.device()));
    torch::Tensor src2 = self_attn(with_pos_embed(src, pos),
                                   reference_pts,
                                   input_flatten,
                                   Spatial_shapes,
                                   input_level_start_index,
                                   input_padding_mask_list);
    src = src + dropout1(src2);
    src = norm1(src);
    src = forward_ffn(src);
    return src;
}


class MSDeformAttnTransformerEncoderOnly : torch::nn::Module {
/*
    Parent class for Transformer Encoder using deformable attention
*/
public:
    MSDeformAttnTransformerEncoderOnly(int d = 256,
                                       int nhead = 8,
                                       const int& num_encoder_layers = 6,
                                       int dim_feedforward = 1024,
                                       float dropout = 0.1,
                                       std::string activation = "relu",
                                       int num_feature_levels = 4,
                                       int enc_n_points = 4)
    : d_model(d), nhead(nhead)
    {
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model,
                                                            dim_feedforward,
                                                            dropout,
                                                            activation,
                                                            num_feature_levels,
                                                            nhead,
                                                            enc_n_points);
        encoder = MSDeformAttnTransformerEncoder(encoder_layer,
                                                 num_encoder_layers);
        register_parameter("level_embed", torch::tensor({num_feature_levels,
                                                         d_model}));
        reset_parameters();
    }
    // class methods:
    void reset_parameters();
    torch::Tensor get_valid_ratio(torch::Tensor&);
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor>&,
                                       std::vector<torch::Tensor>&,
                                       std::vector<torch::Tensor>&);

    int d_model;
    int nhead;
    MSDeformAttnTransformerEncoderLayer encoder_layer;
    MSDeformAttnTransformerEncoder encoder;
    std::vector< std::pair<int, int> > spatial_shapes;
};

void MSDeformAttnTransformerEncoderOnly::reset_parameters() {
    // Parameter regularization/normalization

    for (auto& p : parameters()) {
        if (p.dim() > 1)
            torch::nn::init::xavier_uniform_(p);
    }
    for (auto& m : modules()) {
        if (typeid(m).name() == "MSDeformAttn")
            static_cast<MSDeformAttn>(m)->reset_parameters();
    }
    torch::nn::init::normal_(this->named_parameters()["level_embed"]);
}

torch::Tensor MSDeformAttnTransformerEncoderOnly::get_valid_ratio(torch::Tensor& mask)
{
    double H = double(mask.sizes()[1]);
    double W = double(mask.sizes()[2]);
    torch::Tensor valid_H = torch::sum(~mask.index({torch::indexing::Slice(),
                                                   torch::indexing::Slice(),
                                                   0}),
                                       1);
    torch::Tensor valid_W = torch::sum(~mask.index({torch::indexing::Slice(),
                                                   0,
                                                   torch::indexing::Slice()}),
                                       1);
    torch::Tensor valid_ratio_h = torch::div(valid_H.to(torch::kFloat), H);
    torch::Tensor valid_ratio_w = torch::div(valid_W.to(torch::kFloat), W);
    torch::Tensor valid_ratio = torch::stack({valid_ratio_w, valid_ratio_h}, -1);
    return valid_ratio;
}

std::vector<torch::Tensor> MSDeformAttnTransformerEncoderOnly::forward(
    std::vector<torch::Tensor>& srcs,
    std::vector<torch::Tensor>& masks,
    std::vector<torch::Tensor>& pos_embeds)
{
    /*
        Use this forward pass before decoder-forward to complete a full pass through transformer
    */
    int enable_mask = 0;
    if (!masks.empty()) {
        for (torch::Tensor& src : srcs) {
            if (src.sizes()[2]%32 || src.sizes()[3]%32)
                enable_mask = 1;
        }
    }

    if (enable_mask == 0) {
        for (torch::Tensor& x : srcs) {
            masks.push_back(torch::zeros({x.sizes()[0],
                                          x.sizes()[1],
                                          x.sizes()[2],
                                          x.sizes()[3]},
                                          torch::TensorOptions().device(x.device()).dtype(torch::kBool)));
        }
    }

    std::vector<torch::Tensor> src_flatten{};
    std::vector<torch::Tensor> mask_flatten{};
    std::vector<torch::Tensor> lvl_pos_embed_flatten{};
    for (decltype(srcs.size()) i = 0; i < srcs.size(); ++i) {
        torch::Tensor& src = srcs[i];
        torch::Tensor& mask = masks[i];
        torch::Tensor& pos_embed = pos_embeds[i];
        int h = int(src.sizes()[2]);
        int w = int(src.sizes()[3]);
        spatial_shapes.push_back(std::make_pair(h, w));
        src = src.flatten(2).transpose(1,2);
        torch::Tensor lvl_pos_embed = pos_embed + this->named_parameters()["level_embed"][i].view({1, 1, -1});
        lvl_pos_embed_flatten.push_back(lvl_pos_embed);
        src_flatten.push_back(src);
        mask_flatten.push_back(mask);
    }

    torch::Tensor Src_flatten = torch::cat(src_flatten, 1);
    torch::Tensor Mask_flatten = torch::cat(mask_flatten, 1);
    torch::Tensor Lvl_pos_embed_flatten = torch::cat(lvl_pos_embed_flatten, 1);
    torch::Tensor Spatial_shapes = torch::from_blob(spatial_shapes.data(),
                                                    spatial_shapes.size(),
                                                    torch::TensorOptions().dtype(torch::kLong)
                                                                          .device(Src_flatten.device()));
    torch::Tensor Level_start_index = torch::cat({
        Spatial_shapes.new_zeros((1)),
        Spatial_shapes.prod(1).cumsum(0).index({torch::indexing::Slice(torch::indexing::None, -1)})
    });
    
    std::vector<torch::Tensor> valid_ratios_vector{};
    for (torch::Tensor &m : masks)
        valid_ratios_vector.push_back(m);
    torch::Tensor valid_ratios = torch::stack(valid_ratios_vector, 1);
    torch::Tensor memory = this->encoder->forward(Src_flatten,
                                                  spatial_shapes,
                                                  Level_start_index,
                                                  valid_ratios,
                                                  Lvl_pos_embed_flatten,
                                                  Mask_flatten);
    return std::vector<torch::Tensor> {memory, Level_start_index};
}


void c2_xavier_fill(torch::nn::Conv2d& conv)
{
        torch::nn::init::kaiming_uniform_(conv->weight.data(), 1);
        torch::nn::init::constant_(conv->bias.data(), 0);
}


class MaskDINO2Encoder : torch::nn::Module {
/*
    This is the multi-scale encoder in detection models,
    also named as pixel decoder in segmentation models.
*/
public:
    MaskDINO2Encoder(std::set< std::pair<std::string, ShapeSpec> >&,
                     float const&,
                     int const&,
                     int const&,
                     int const&,
                     int const&,
                     int const&,
                     std::string const&,
                     std::vector<std::string>&,
                     int const&,
                     int const&,
                     int const&,
                     std::string const&);
    // class methods:
    std::vector<torch::Tensor> forward_features(std::unordered_map<std::string,
                                                torch::Tensor>&,
                                                std::vector<torch::Tensor>&);
private:
    std::vector<std::string>
        in_features,
        transformer_in_features;
    std::vector<int>
        feature_strides,
        feature_channels,
        transformer_in_channels,
        transformer_feature_strides;
    std::string
        feature_order;
    int64_t
        low_resolution_index;
    int
        maskdino_num_feature_levels,
        common_stride,
        transformer_num_feature_levels,
        high_resolution_index,
        mask_dim,
        num_fpn_levels,
        total_num_feature_levels;
    MSDeformAttnTransformerEncoderOnly
        transformer;
    PositionEmbeddingSine
        pe_layer;
    torch::nn::Conv2d
        mask_features = nullptr;
    torch::nn::ModuleList
        input_proj = nullptr,
        lateral_convs = nullptr,
        output_convs = nullptr;
};

MaskDINO2Encoder::MaskDINO2Encoder(
    std::set< std::pair<std::string, ShapeSpec> >& input_shape,
    float const& transformer_dropout,
    int const& transformer_nheads,
    int const& transformer_dim_feedforward,
    int const& transformer_enc_layers,
    int const& conv_dim,
    int const& mask_dim,
    std::string const& norm,
    std::vector<std::string>& transformer_in_features_local,
    int const& common_stride,
    int const& num_feature_levels,
    int const& total_num_feature_levels,
    std::string const& feature_order = "low2high"
) : torch::nn::Module(),
    feature_order(feature_order), 
    total_num_feature_levels(total_num_feature_levels),
    common_stride(common_stride)
{
/*
    Args:
        input_shape: shapes (channels and stride) of the input features
        transformer_dropout: dropout probability in transformer
        transformer_nheads: number of heads in transformer
        transformer_dim_feedforward: dimension of feedforward network
        transformer_enc_layers: number of transformer encoder layers
        conv_dims: number of output channels for the intermediate conv layers.
        mask_dim: number of output channels for the final conv layer.
        norm (str or callable): normalization for all conv layers
        num_feature_levels: feature scales used
        total_num_feature_levels: total feautre scales used (include the downsampled features)
        feature_order: 'low2high' or 'high2low', i.e., 'low2high' means low-resolution features are put in the first.
*/
    std::set< std::pair<std::string, ShapeSpec> > transformer_input_shape{};
    for (const auto& [k, v] : input_shape) {
        in_features.push_back(k);
        feature_strides.push_back(v.stride);
        feature_channels.push_back(v.channels);
        if (std::find(transformer_in_features_local.begin(),
                      transformer_in_features_local.end(),
                      k)
            != transformer_in_features_local.end())
        {
            transformer_input_shape.insert(std::make_pair(k, v));
        }
    }
    std::sort(transformer_input_shape.begin(), transformer_input_shape.end(),
        [=](const auto& a){ return a.second.stride; }
    );
    input_shape = transformer_input_shape;
    if (feature_order == "low2high") {
        struct {
            bool operator()(std::pair<std::string, ShapeSpec>a, std::pair<std::string, ShapeSpec>b)
                const { return a.second.stride < b.second.stride; }            
        } strideLess;
        std::sort(transformer_input_shape.begin(), transformer_input_shape.end(), strideLess);
    }
    else {
        struct {
            bool operator()(std::pair<std::string, ShapeSpec>a, std::pair<std::string, ShapeSpec>b)
                const { return a.second.stride > b.second.stride; }            
        } strideGreater;
        std::sort(transformer_input_shape.begin(), transformer_input_shape.end(), strideGreater);
    }
    for (const auto& [k, v] : transformer_input_shape) {
        this->transformer_in_features.push_back(k);
        this->transformer_in_channels.push_back(v.channels);
        this->transformer_feature_strides.push_back(v.stride);
    }
    this->maskdino_num_feature_levels = num_feature_levels;
    transformer_num_feature_levels = int(this->transformer_in_features.size());
    this->low_resolution_index =
        std::distance(transformer_in_channels.begin(),
                      std::max_element(transformer_in_channels.begin(),
                                       transformer_in_channels.end()));
    if (this->feature_order == "low2high")
        this->high_resolution_index = 0;
    else
        this->high_resolution_index = -1;    
    if (this->transformer_num_feature_levels > 1) {
        this->input_proj();
        for (auto in_channels = transformer_in_channels.end(); in_channels > transformer_in_channels.begin(); --in_channels) {
            torch::nn::Conv2d conv = torch::nn::Conv2d(*in_channels, conv_dim, 1);
            torch::nn::init::xavier_uniform_(conv->weight.data());
            torch::nn::init::constant_(conv->bias.data(), 0);
            input_proj->push_back(torch::nn::Sequential(
                conv,
                torch::nn::GroupNorm(32, conv_dim)
            ));
        }
        int in_channels = *std::max_element(transformer_in_channels.begin(), transformer_in_channels.end());
        for (int i = this->transformer_num_feature_levels; i < this->total_num_feature_levels; ++i) {
            torch::nn::Conv2d conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, conv_dim, 3).stride(2).padding(1));
            torch::nn::init::xavier_uniform_(conv->weight.data());
            torch::nn::init::constant_(conv->bias.data(), 0);
            input_proj->push_back(torch::nn::Sequential(
                conv,
                torch::nn::GroupNorm(32, conv_dim)
            ));
            in_channels = conv_dim;
        }
    } 
    else {
        this->input_proj();
        torch::nn::Conv2d conv = torch::nn::Conv2d(transformer_in_channels[-1], conv_dim, 1);
        torch::nn::init::xavier_uniform_(conv->weight.data());
        torch::nn::init::constant_(conv->bias.data(), 0);
        input_proj->push_back(torch::nn::Sequential(
            conv,
            torch::nn::GroupNorm(32, conv_dim)
        ));
    }
    this->transformer = MSDeformAttnTransformerEncoderOnly(conv_dim,
                                                           transformer_nheads,
                                                           transformer_enc_layers,
                                                           transformer_dim_feedforward,
                                                           transformer_dropout,
                                                           "relu",
                                                           this->total_num_feature_levels);
    int N_steps = int(conv_dim / 2);
    this->pe_layer = PositionEmbeddingSine(N_steps, 10000, true);

    this->mask_dim = mask_dim;
    this->mask_features = torch::nn::Conv2d(torch::nn::Conv2dOptions(conv_dim, mask_dim, 1).stride(1).padding(0));
    c2_xavier_fill(this->mask_features);
    int stride = *min_element(transformer_feature_strides.begin(), transformer_feature_strides.end());
    this->num_fpn_levels = std::max(int(std::log2(stride) - std::log2(this->common_stride)), 1);

    std::vector<torch::nn::Sequential> lateral_convs_local, output_convs_local;

    bool use_bias = norm.empty();
    int idx = 1;
    for (auto in_channels = feature_channels.begin(); in_channels < feature_channels.begin() + num_fpn_levels; ++in_channels) {
        torch::nn::Conv2d lateral_conv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(*in_channels, conv_dim, 3).stride(1).padding(1).bias(use_bias)
        );
        torch::nn::Conv2d output_conv = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(conv_dim, conv_dim, 3).stride(1).padding(1).bias(use_bias)
        );
        c2_xavier_fill(lateral_conv);
        c2_xavier_fill(output_conv);
        torch::nn::Sequential adapter = torch::nn::Sequential(lateral_conv);
        torch::nn::Sequential layer = torch::nn::Sequential(output_conv);
        if (!use_bias) {
            if (norm == "BN") {
                adapter->push_back(torch::nn::BatchNorm2d(conv_dim));
                layer->push_back(torch::nn::BatchNorm2d(conv_dim));
            }
            else if ("GN") {
                adapter->push_back(torch::nn::GroupNorm(32, conv_dim));
                layer->push_back(torch::nn::GroupNorm(32, conv_dim));
            }
            else if ("LN") {
                adapter->push_back(torch::nn::LayerNorm(conv_dim));
                layer->push_back(torch::nn::LayerNorm(conv_dim));
            }
        }
        adapter->push_back(torch::nn::ReLU());
        layer->push_back(torch::nn::ReLU());
        
        this->register_module("adapter_" + idx, adapter);
        this->register_module("layer_" + idx, layer);
        
        lateral_convs_local.push_back(adapter);
        output_convs_local.push_back(layer);

        ++idx;
    }
    this->lateral_convs = torch::nn::ModuleList(lateral_convs_local);
    this->output_convs = torch::nn::ModuleList(output_convs_local);
}


std::vector<torch::Tensor> MaskDINO2Encoder::forward_features(
    std::unordered_map<std::string, torch::Tensor>& features,
    std::vector<torch::Tensor>& masks
)
{
/*
    :param features: multi-scale features from the backbone
    :param masks: image mask
    :return: enhanced multi-scale features and mask feature (1/4 resolution) for the decoder to produce binary mask
*/
    std::vector<torch::Tensor> srcs;
    std::vector<torch::Tensor> pos;
    std::vector<torch::Tensor> srcsl;
    std::vector<torch::Tensor> posl;

    if (total_num_feature_levels > transformer_num_feature_levels) {
        torch::Tensor smallest_feat = features[transformer_in_features[low_resolution_index]].to(torch::kFloat);
        int len_src = transformer_num_feature_levels;
        for (int i = len_src; i < total_num_feature_levels; ++i) {
            torch::Tensor src;
            if (i == len_src)
                src = input_proj[size_t(i)]->as<torch::nn::Sequential>()->forward(smallest_feat);
            else
                src = input_proj[size_t(i)]->as<torch::nn::Sequential>()->forward(srcsl[-1]);
            srcsl.push_back(src);
            posl.push_back(pe_layer.forward(src));
        }
    }
    std::reverse(srcsl.begin(), srcsl.end());

    size_t idx = 0;
    for (auto f = transformer_in_features.end(); f > transformer_in_features.begin(); --f) {
        torch::Tensor x = features[*f].to(torch::kFloat);
        srcs.push_back(input_proj[idx]->as<torch::nn::Sequential>()->forward(x));
        pos.push_back(pe_layer.forward(x));
        ++idx;
    }
    if (feature_order == "low2high") {
        srcs.insert(srcs.end(), srcsl.begin(), srcsl.end());
        pos.insert(pos.end(), posl.begin(), posl.end());
    }
    else {
        srcsl.insert(srcsl.end(), srcs.begin(), srcs.end());
        posl.insert(posl.end(), pos.begin(), pos.end());
        srcs = srcsl;
        pos = posl;
    }

    std::vector<torch::Tensor> output = this->transformer.forward(srcs, masks, pos);
    torch::Tensor& y = output[0];
    torch::Tensor& level_start_index = output[1];
    std::vector<std::pair<int, int> > &spatial_shapes = transformer.spatial_shapes;
    int bs = int(y.sizes()[0]);
    
    std::vector<int64_t> split_size_or_sections{};
    for (int i = 0; i < this->total_num_feature_levels; ++i) {
        if (i < this->total_num_feature_levels - 1) {
            int64_t val = level_start_index[i+1].data_ptr<int64_t>() - level_start_index[i].data_ptr<int64_t>();
            split_size_or_sections.push_back(val);
        }
        else {
            int64_t val = int64_t(y.sizes()[1]) - int64_t(level_start_index[i].data_ptr<int64_t>());
            split_size_or_sections.push_back(val);
        }
    }
    std::vector<torch::Tensor> y_vec = torch::split(y, c10::IntArrayRef(split_size_or_sections), 1);
    
    std::vector<torch::Tensor> out{}, multiscale_features{};
    int num_cur_levels = 0;
    for (decltype(y_vec.size()) i = 0; i < y_vec.size(); ++i) {
        out.push_back(y_vec[i].transpose(1, 2).view({bs, -1, std::get<0>(spatial_shapes[i]), std::get<1>(spatial_shapes[i])}));
    }

    int idx_int = 0;
    for (auto ptr = (in_features.begin() + num_fpn_levels); ptr > in_features.begin(); --ptr) {
        torch::Tensor x = features[*ptr].to(torch::kFloat);
        torch::nn::Conv2dImpl *lateral_conv = this->lateral_convs[idx_int]->as<torch::nn::Conv2d>();
        torch::nn::Conv2dImpl *output_conv = this->output_convs[idx_int]->as<torch::nn::Conv2d>();
        torch::Tensor cur_fpn = (*lateral_conv).forward(x);
        std::vector<int64_t> sizes{cur_fpn.sizes()[-2], cur_fpn.sizes()[-1]};
        y = cur_fpn + torch::nn::functional::interpolate(out[this->high_resolution_index],
                                                         torch::nn::functional::InterpolateFuncOptions()
                                                         .size(sizes)
                                                         .mode(torch::kBilinear)
                                                         .align_corners(false));
        y = (*output_conv).forward(y);
        out.push_back(y);
        ++ptr;
    }
    for (auto& o : out) {
        if (num_cur_levels < this->total_num_feature_levels) {
            multiscale_features.push_back(o);
            ++num_cur_levels;
        }
    }
    std::vector<torch::Tensor> ret{this->mask_features(out[-1]), out[0]};
    ret.insert(ret.end(), multiscale_features.begin(), multiscale_features.end());
    return ret;    
}
