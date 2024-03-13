#include <array>
#include <cassert>
#include <math.h>
#include <torch/torch.h>
#include <vector>
#include "detectron2.h"
#include "utils/modules.h"
#include "utils/position_encoding.h"
#ifndef MASKDINO2ENCODER_H
#define MASKDINO2ENCODER_H


class MSDeformAttnTransformerEncoderImpl : torch::nn::Module {
public:
    //default constructor for initialization of classes later in this file
    MSDeformAttnTransformerEncoderImpl();
    MSDeformAttnTransformerEncoderImpl(torch::nn::AnyModule& encoder_layer,
                                       int const& num_l)
        : num_layers(num_l), torch::nn::Module()
        { layers = get_clones(encoder_layer, num_layers); }
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
    torch::nn::Sequential layers = nullptr;
    int num_layers;
};
TORCH_MODULE(MSDeformAttnTransformerEncoder);


torch::Tensor ms_deform_attn_core_pytorch(torch::Tensor& value,
                                          torch::Tensor& value_spatial_shapes,
                                          torch::Tensor& sampling_locations,
                                          torch::Tensor& attention_weights);


class MSDeformAttnImpl : torch::nn::Module {
public:
    MSDeformAttnImpl();
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
    int d_model;
    int n_levels;
    int n_heads;
    int n_points;
    int im2col_step = 128;
    torch::nn::Linear sampling_offsets = nullptr;
    torch::nn::Linear attention_weights = nullptr;
    torch::nn::Linear value_proj = nullptr;
    torch::nn::Linear output_proj = nullptr;
};
TORCH_MODULE(MSDeformAttn);


class MSDeformAttnTransformerEncoderLayer : torch::nn::Module {
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
                                        torch::Tensor& pos);

    torch::Tensor forward_ffn(torch::Tensor& src);

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


class MSDeformAttnTransformerEncoderOnly : torch::nn::Module {
public:
    MSDeformAttnTransformerEncoderOnly(int,
                                       int,
                                       const int&,
                                       int,
                                       float,
                                       std::string,
                                       int,
                                       int);
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


void c2_xavier_fill(torch::nn::Conv2d& conv);

class MaskDINO2Encoder : torch::nn::Module {
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

#endif
