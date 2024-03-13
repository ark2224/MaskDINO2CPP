#include <iostream>
#include <limits>
#include <math.h>
#include <torch/torch.h>
#include "utils/modules.h"
#include "MaskDINO2Encoder.h"


torch::nn::Sequential _get_clones(DeformableTransformerDecoderLayer module, int N);


class DeformableTransformerDecoderLayerImpl : torch::nn::Module {
/*
    Single layer class for Decoder
*/
public:
    DeformableTransformerDecoderLayerImpl();
    DeformableTransformerDecoderLayerImpl(int,
                                          int,
                                          float,
                                          std::string,
                                          int,
                                          int,
                                          int,
                                          std::string);
    // class methods:
    static torch::Tensor with_pos_embed(torch::Tensor x, torch::Tensor pos);
    torch::Tensor forward_ffn(torch::Tensor&);
    torch::Tensor forward(torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor,
                          torch::Tensor);
private:
    int
        d_model,
        d_ffn,
        n_levels,
        n_heads,
        n_points;
    float
        dropout;
    MSDeformAttn
        cross_attn;
    torch::nn::MultiheadAttention*
        self_attn = nullptr;
    torch::nn::Dropout
        dropout1,
        dropout2,
        dropout3,
        dropout4;
    torch::nn::LayerNorm
        norm1,
        norm2,
        norm3;
    torch::nn::Linear 
        linear1,
        linear2;
    torch::nn::Sequential
        activation_layer;
    std::string
        key_aware_type;

};
TORCH_MODULE(DeformableTransformerDecoderLayer);


class TransformerDecoder : torch::nn::Module {
public:
    TransformerDecoder();
    TransformerDecoder(DeformableTransformerDecoderLayer&,
                       int&,
                       torch::nn::LayerNorm&,
                       bool,
                       int,
                       int,
                       bool,
                       int,
                       bool,
                       std::vector<DeformableTransformerDecoderLayer>,
                       bool,
                       std::vector<float>);
    // class methods:
    void reset_parameters();
    std::vector< std::vector<torch::Tensor> > forward(torch::Tensor&,
                                                      torch::Tensor&,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor,
                                                      torch::Tensor);

    int num_layers,
        query_dim,
        num_feature_levels,
        d_model;
    std::vector<float>
        dec_layer_dropout_prob;
    torch::nn::LayerNorm
        norm;
    bool 
        return_intermediate,
        modulate_attention,
        deformable_decoder;
    MLPImpl
        *ref_point_head = nullptr,
        *query_pos_sine_scale = nullptr,
        *query_scale = nullptr,
        *ref_anchor_head = nullptr;
    std::vector<DeformableTransformerDecoderLayer>
        decoder_layer_number;
    torch::nn::Sequential
        layers,
        bbox_embed;
};


class MaskDINO2Decoder : torch::nn::Module {
public:
    MaskDINO2Decoder();
    MaskDINO2Decoder(int,
                     int,
                     int,
                     int,
                     int,
                     int,
                     int,
                     int,
                     bool,
                     bool,
                     std::string,
                     float,
                     int,
                     std::string,
                     bool,
                     bool,
                     bool,
                     int,
                     float,
                     std::string,
                     int,
                     int,
                     bool,
                     int,
                     bool,
                     bool);
    // class methods:
    std::vector<torch::Tensor> prepare_for_dn(std::vector< std::unordered_map<std::string, torch::Tensor> >&,
                                              torch::Tensor&,
                                              torch::Tensor&,
                                              int);
    std::vector<torch::Tensor> dn_post_process(torch::Tensor&,
                                  torch::Tensor&,
                                  std::unordered_map<std::string, torch::Tensor>&,
                                  torch::Tensor&);
    torch::Tensor get_valid_ratio(torch::Tensor&);
    torch::Tensor pred_box(std::vector<torch::Tensor>&,
                           std::vector<torch::Tensor>&,
                           torch::Tensor);
    std::unordered_map<std::string, torch::Tensor> forward(std::vector<torch::Tensor>&,
                                                           torch::Tensor&,
                                                           std::vector<torch::Tensor>&,
                                                           std::vector< std::unordered_map<std::string, torch::Tensor> >&);
    std::vector<torch::Tensor> forward_prediction_heads(torch::Tensor&,
                                                        torch::Tensor&,
                                                        bool);
    std::unordered_map<std::string, torch::Tensor> _set_aux_loss(torch::Tensor&,
                                                                 torch::Tensor&,
                                                                 torch::Tensor);

private:
    bool 
        mask_classification,
        learn_tgt,
        initial_pred,
        semantic_ce_loss,
        two_stage;
    int
        num_feature_levels,
        hidden_dim,
        num_queries,
        num_heads,
        num_classes,
        num_layers,
        total_num_feature_levels,
        dn_num;
    float
        noise_scale;
    std::string
        initialize_box_type,
        dn;
    torch::nn::Sequential
        input_proj,
        bbox_embed;
    torch::nn::Embedding
        query_feat,
        query_embed,
        label_enc;
    MLPImpl
        mask_embed,
        _bbox_embed;
    torch::nn::Linear
        class_embed,
        enc_output;
    torch::nn::LayerNorm
        enc_output_norm,
        decoder_norm;
    TransformerDecoder
        decoder;
    std::unordered_map<std::string, torch::Tensor>
        mask_dict_inclass;
};
