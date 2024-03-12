#include <iostream>
#include <torch/torch.h>
#include <math.h>
#include <limits>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "include/structures/image_list.h"
#include "memory.h"
#include "SwinL.cpp"
#include "include/utils/box_operations.h"
#include "MaskDINO2Encoder.cpp"


torch::Tensor gen_sineembed_for_position(torch::Tensor pos_tensor) {
    double scale = 2 * M_PI;
    torch::Tensor dim_t = torch::arange(128, torch::TensorOptions().dtype(torch::kFloat32).device(pos_tensor.device()));
    dim_t = pow(10000, 2 * (floor_divide(dim_t, 2)) / 128);
    torch::Tensor x_embed = pos_tensor.index({torch::indexing::Slice(),
                                              torch::indexing::Slice(),
                                              0})
                            * scale;
    torch::Tensor y_embed = pos_tensor.index({torch::indexing::Slice(),
                                              torch::indexing::Slice(),
                                              1})
                            * scale;
    torch::Tensor pos_x = x_embed.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::None})
                          / dim_t;
    torch::Tensor pos_y = y_embed.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::None})
                          / dim_t;
    pos_x = torch::stack((pos_x.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice(0, torch::indexing::None, 2)
                                       }).sin(),
                          pos_x.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice(1, torch::indexing::None, 2)
                                       }).cos()),
                          3).flatten(2);
    pos_y = torch::stack((pos_y.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice(0, torch::indexing::None, 2)
                                       }).sin(),
                          pos_y.index({torch::indexing::Slice(),
                                       torch::indexing::Slice(),
                                       torch::indexing::Slice(1, torch::indexing::None, 2)
                                       }).cos()),
                          3).flatten(2);
    torch::Tensor pos;
    if (pos_tensor.sizes()[-1] == 2)
        pos = torch::cat((pos_y, pos_x), 2);
    else if (pos_tensor.sizes()[-1] == 4) {
        torch::Tensor w_embed = pos_tensor.index({torch::indexing::Slice(),
                                                  torch::indexing::Slice(),
                                                  2})
                                * scale;
        torch::Tensor pos_w = w_embed.index({torch::indexing::Slice(),
                                             torch::indexing::Slice(),
                                             torch::indexing::None})
                              / dim_t;
        pos_w = torch::stack((pos_w.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(0, torch::indexing::None, 2)}).sin(),
                              pos_w.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(1, torch::indexing::None, 2)}).cos()),
                              3).flatten(2);
        torch::Tensor h_embed = pos_tensor.index({torch::indexing::Slice(),
                                                  torch::indexing::Slice(),
                                                  3})
                                * scale;
        torch::Tensor pos_h = h_embed.index({torch::indexing::Slice(),
                                             torch::indexing::Slice(),
                                             torch::indexing::None})
                              / dim_t;
        pos_h = torch::stack((pos_h.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(0, torch::indexing::None, 2)}).sin(),
                              pos_h.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(1, torch::indexing::None, 2)}).cos()),
                              3).flatten(2);
        torch::Tensor pos = torch::cat((pos_y, pos_x, pos_w, pos_h), 2);
    }
    else
        std::cerr << "Unknown pos_tensor shape(-1): " << pos_tensor.sizes()[-1];
    
    return pos;
}


torch::Tensor inverse_sigmoid(torch::Tensor x, double eps = 1e-5) {
    x = x.clamp(0, 1);
    torch::Tensor x1 = x.clamp(eps);
    torch::Tensor x2 = (1 - x).clamp(eps);
    return torch::log(x1 / x2);
}


std::vector<torch::Tensor> gen_encoder_output_proposals(torch::Tensor& memory, torch::Tensor& memory_padding_mask, std::vector< std::pair<int, int> >& spatial_shapes) {
    int N_ = (int)memory.sizes()[0];
    int S_ = (int)memory.sizes()[1];
    int C_ = (int)memory.sizes()[2];
    float base_scale = 4.0;
    std::vector<torch::Tensor> proposals{};
    int _cur = 0, idx = 0;
    for (auto& shape : spatial_shapes) {
        int H_ = shape.first; 
        int W_ = shape.second;
        torch::Tensor mask_flatten_ = memory_padding_mask.index(
                                          {torch::indexing::Slice(),
                                          torch::indexing::Slice(_cur, _cur + H_ + W_)}
                                      ).view({N_, H_, W_, 1});
        torch::Tensor valid_H = torch::sum(~mask_flatten_.index({torch::indexing::Slice(),
                                                                 torch::indexing::Slice(),
                                                                 0,
                                                                 0}),
                                           1);
        torch::Tensor valid_W = torch::sum(~mask_flatten_.index({torch::indexing::Slice(),
                                                                 0,
                                                                 torch::indexing::Slice(),
                                                                 0}),
                                           1);
        std::vector<torch::Tensor> grids_xy = torch::meshgrid(
            {torch::linspace(0, H_ - 1, H_, torch::TensorOptions().dtype(torch::kFloat32).device(memory.device())),
            torch::linspace(0, W_ - 1, W_, torch::TensorOptions().dtype(torch::kFloat32).device(memory.device()))}
        );
        torch::Tensor grid_x = grids_xy[1],
                      grid_y = grids_xy[0];
        torch::Tensor grid = torch::cat({grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)}, -1);

        torch::Tensor scale = torch::cat({valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)}, 1).view({N_, 1, 1, 2});
        grid = (grid.unsqueeze(0).expand({N_, -1, -1, -1}) + 0.5) / scale;
        torch::Tensor wh = torch::ones_like(grid) * 0.05 * (pow(2.0, idx));
        torch::Tensor proposal = torch::cat({grid, wh}, -1).view({N_, -1, 4});
        proposals.push_back(proposal);
        _cur += H_ * W_;
        ++idx;
    }
    torch::Tensor output_proposals = torch::cat(proposals, 1);
    torch::Tensor output_proposals_valid = torch::logical_and((output_proposals > 0.01), (output_proposals < 0.99)).all(-1, true);
    output_proposals = torch::log(output_proposals / (1 - output_proposals));
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), std::numeric_limits<float>::infinity());
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, std::numeric_limits<float>::infinity());

    torch::Tensor output_memory = memory;
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0));
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0));
    return std::vector<torch::Tensor> {output_memory, output_proposals};
}


class DeformableTransformerDecoderLayerImpl : torch::nn::Module {
public:
    DeformableTransformerDecoderLayerImpl() : torch::nn::Module() { }
    DeformableTransformerDecoderLayerImpl(int,
                                      int,
                                      float,
                                      std::string,
                                      int,
                                      int,
                                      int,
                                      std::string);
    // class methods:
    static torch::Tensor with_pos_embed(torch::Tensor x, torch::Tensor pos) 
    {
        if (pos.defined())
            return x;
        else
            return x + pos;
    }
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

DeformableTransformerDecoderLayerImpl::DeformableTransformerDecoderLayerImpl(
    int d_model = 256,
    int d_ffn = 1024,
    float dropout = 0.1,
    std::string activation = "relu",
    int n_levels = 4,
    int n_heads = 8,
    int n_points = 4,
    std::string key_aware_type = ""
) : torch::nn::Module(), key_aware_type(key_aware_type)
{
    this->cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points);
    dropout1 = torch::nn::Dropout(dropout);
    norm1 = torch::nn::LayerNorm(d_model);

    self_attn = &torch::nn::MultiheadAttention(d_model, n_heads, dropout);
    dropout2 = torch::nn::Dropout(dropout);
    norm2 = torch::nn::LayerNorm(d_model);

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
    dropout3 = torch::nn::Dropout(dropout);
    linear2 = torch::nn::Linear(d_ffn, d_model);
    dropout4 = torch::nn::Dropout(dropout);
    norm3 = torch::nn::LayerNorm(d_model);
}


torch::Tensor DeformableTransformerDecoderLayerImpl::forward_ffn(torch::Tensor& tgt) {
    torch::Tensor tgt2 = linear2(dropout3(activation_layer->forward(linear1(tgt))));
    tgt = tgt + dropout4(tgt);
    tgt = norm3(tgt);
    return tgt;
}


torch::Tensor DeformableTransformerDecoderLayerImpl::forward(
    torch::Tensor tgt = torch::Tensor(),
    torch::Tensor tgt_query_pos = torch::Tensor(),
    torch::Tensor tgt_query_sine_embed = torch::Tensor(),
    torch::Tensor tgt_key_padding_mask = torch::Tensor(),
    torch::Tensor tgt_reference_points = torch::Tensor(),
    torch::Tensor memory = torch::Tensor(),
    torch::Tensor memory_key_padding_mask = torch::Tensor(),
    torch::Tensor memory_level_start_index = torch::Tensor(),
    torch::Tensor memory_spatial_shapes = torch::Tensor(),
    torch::Tensor memory_pos = torch::Tensor(),
    torch::Tensor self_attn_mask = torch::Tensor(),
    torch::Tensor cross_attn_mask = torch::Tensor())
{
    torch::Tensor tgt2;
    if (self_attn != nullptr) {
        torch::Tensor q, k;
        q = with_pos_embed(tgt, tgt_query_pos);
        k = q;
        tgt2 = std::get<0>((*self_attn)(q, k, tgt, self_attn_mask));
        tgt = tgt + dropout2(tgt2);
        tgt = norm2(tgt);
    }
    if (key_aware_type.empty()) {
        if (key_aware_type == "mean")
            tgt = tgt + memory.mean(0, true);
        else
            std::cerr << "Unknown key_aware_type";
    }

    torch::Tensor arg1 = with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                  arg2 = tgt_reference_points.transpose(0, 1).contiguous(),
                  arg3 = memory.transpose(0, 1);
    std::initializer_list<torch::Tensor> arg4{memory_key_padding_mask};
    tgt2 = cross_attn->forward(arg1,
                      arg2,
                      arg3,
                      memory_spatial_shapes,
                      memory_level_start_index,
                      arg4).transpose(0, 1); 
    
    tgt = tgt + dropout1(tgt2);
    tgt = norm1(tgt);
    tgt = forward_ffn(tgt);

    return tgt;
}


torch::nn::Sequential _get_clones(DeformableTransformerDecoderLayer module, int N) { // overloaded of MaskDINO2Encoder::get_clones
    torch::nn::Sequential ret = {};
    for (int i = 0; i < N; ++N)
        ret->push_back(module);
    return ret;
}


class TransformerDecoder : torch::nn::Module {
public:
    TransformerDecoder() {  }
    TransformerDecoder(DeformableTransformerDecoderLayer&,
                       int&,
                       torch::nn::LayerNorm&,
                       bool,
                       int,
                       int,
                       bool,
                       int,
                       bool,
                    //    decoder_query_perturber,
                       std::vector<DeformableTransformerDecoderLayer>,
                       bool,
                    //    bool,
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

TransformerDecoder::TransformerDecoder(DeformableTransformerDecoderLayer& decoder_layer,
                                       int& num_layers,
                                       torch::nn::LayerNorm& norm,
                                       bool return_intermediate = false,
                                       int d_model = 256,
                                       int query_dim = 4,
                                       bool modulate_hw_attn = true,
                                       int num_feature_levels = 1,
                                       bool deformable_decoder = true,
                                       std::vector<DeformableTransformerDecoderLayer> dec_layer_num = {},
                                       bool rm_dec_query_scale = true,
                                    //    bool dec_layer_share = false,
                                       std::vector<float> dec_layer_dropout_prob = {}) :
torch::nn::Module(), num_layers(num_layers), norm(norm), return_intermediate(return_intermediate),
query_dim(query_dim), num_feature_levels(num_feature_levels), d_model(d_model),
modulate_attention(modulate_hw_attn), deformable_decoder(deformable_decoder),
decoder_layer_number(dec_layer_num), dec_layer_dropout_prob(dec_layer_dropout_prob)
{
    if (num_layers > 0) {
        layers = _get_clones(decoder_layer, num_layers);
        register_module("layers", layers);
    } else
        layers = {};
    assert(return_intermediate);
    ref_point_head = &MLPImpl(int(query_dim / 2) * d_model, d_model, d_model, 2);
    if (!deformable_decoder)
        query_pos_sine_scale = &MLPImpl(d_model, d_model, d_model, 2);
    if (!rm_dec_query_scale)
        std::cerr << "Need to implement.";
        query_scale = &MLPImpl(d_model, d_model, d_model, 2);
    if (!deformable_decoder && modulate_hw_attn)
        ref_anchor_head = &MLPImpl(d_model, d_model, 2, 2);
    if (!dec_layer_num.empty()) {
        assert(typeid(dec_layer_num).name() == "std::vector<DeformableTransformerDecoderLayer>");
        assert(dec_layer_num.size() == num_layers);
    }
    if (!dec_layer_dropout_prob.empty()) {
        assert(typeid(dec_layer_dropout_prob).name() == "std::vector<float>");
        assert(dec_layer_dropout_prob.size() == num_layers);
        for (auto& prob : dec_layer_dropout_prob)
            assert(prob >= 0.0 && prob <= 1.0);
    }

    reset_parameters();
}


std::vector< std::vector<torch::Tensor> > TransformerDecoder::forward(
    torch::Tensor& tgt,
    torch::Tensor& memory,
    torch::Tensor tgt_mask = torch::Tensor(),
    torch::Tensor memory_mask = torch::Tensor(),
    torch::Tensor tgt_key_padding_mask = torch::Tensor(),
    torch::Tensor memory_key_padding_mask = torch::Tensor(),
    torch::Tensor pos = torch::Tensor(),
    torch::Tensor refpoints_unsigmoid = torch::Tensor(),
    torch::Tensor level_start_index = torch::Tensor(),
    torch::Tensor spatial_shapes = torch::Tensor(),
    torch::Tensor valid_ratios = torch::Tensor()) 
{
    torch::Tensor output = tgt;
    torch::Device device = tgt.device();
    std::vector<torch::Tensor> intermediate{};
    torch::Tensor reference_points = refpoints_unsigmoid.sigmoid().to(device);
    std::vector<torch::Tensor> ref_points = {reference_points};

    int idx = 0;
    for (auto layer = layers->begin(); layer != layers->end(); ++layer) {
        torch::Tensor reference_points_input = 
            reference_points.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(),
                                    torch::indexing::None})
            * torch::cat({valid_ratios, valid_ratios}, -1).index({torch::indexing::None,
                                                                  torch::indexing::Slice()});
        torch::Tensor query_sine_embed = 
            gen_sineembed_for_position(reference_points_input.index({torch::indexing::Slice(),
                                                                     torch::indexing::Slice(),
                                                                     0,
                                                                     torch::indexing::Slice()}));
        torch::Tensor raw_query_pos = ref_point_head->forward(query_sine_embed);
        torch::Tensor pos_scale;
        if (query_scale != nullptr)
            pos_scale = query_scale->forward(output);
        else
            pos_scale = torch::ones(1);
        torch::Tensor query_pos = pos_scale * raw_query_pos;
        torch::Tensor output = layer->forward(output,
                                              query_pos,
                                              query_sine_embed,
                                              tgt_key_padding_mask,
                                              reference_points_input,
                                              memory,
                                              memory_key_padding_mask,
                                              level_start_index,
                                              spatial_shapes,
                                              pos,
                                              tgt_mask,
                                              memory_mask);

        if (!bbox_embed.is_empty()) {
            torch::Tensor reference_before_sigmoid = inverse_sigmoid(reference_points);
            torch::Tensor delta_unsig = bbox_embed->forward(output).to(device);
            torch::Tensor outputs_unsig = delta_unsig + reference_before_sigmoid;
            torch::Tensor new_reference_points = outputs_unsig.sigmoid();

            torch::Tensor reference_points = new_reference_points.detach();
            ref_points.push_back(new_reference_points);
        }
        intermediate.push_back(norm(output));
        ++idx;
    }
    
    std::vector< std::vector<torch::Tensor> > ret{};
    std::vector<torch::Tensor> ret_1{}, ret_2{};
    for (torch::Tensor& itm_out : intermediate) {
        ret_1.push_back(itm_out.transpose(0, 1));
    }
    for (torch::Tensor& itm_refpoint : ref_points) {
        ret_2.push_back(itm_refpoint.transpose(0, 1));
    }
    ret.push_back(ret_1);
    ret.push_back(ret_2);
    return ret;
}


class MaskDINO2Decoder : torch::nn::Module {
public:
    MaskDINO2Decoder() { }
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

MaskDINO2Decoder::MaskDINO2Decoder(
    int in_channels,
    int num_classes,
    int hidden_dim,
    int num_queries,
    int nheads,
    int dim_feedforward,
    int dec_layers,
    int mask_dim,
    bool enforce_input_project,
    bool two_stage,
    std::string dn,
    float noise_scale,
    int dn_num,
    std::string initialize_box_type,
    bool initial_pred,
    bool learn_tgt,
    bool mask_classification = true,
    int total_num_feature_levels = 4,
    float dropout = 0.0,
    std::string activation = "relu",
    int nhead = 8,
    int dec_n_points = 4,
    bool return_imediate_dec = true,
    int query_dim = 4,
    bool dec_layer_share = false,
    bool semantic_ce_loss = false) :
torch::nn::Module(),
mask_classification(mask_classification),
num_feature_levels(total_num_feature_levels),
initial_pred(initial_pred),
dn(dn),
learn_tgt(learn_tgt),
noise_scale(noise_scale),
dn_num(dn_num),
num_heads(nheads),
num_layers(dec_layers),
two_stage(two_stage),
initialize_box_type(initialize_box_type),
total_num_feature_levels(total_num_feature_levels),
num_queries(num_queries),
semantic_ce_loss(semantic_ce_loss),
num_classes(num_classes),
hidden_dim(hidden_dim)
{
    /*NOTE: this interface is experimental.
    Args:
        in_channels: channels of the input features
        mask_classification: whether to add mask classifier or not
        num_classes: number of classes
        hidden_dim: Transformer feature dimension
        num_queries: number of queries
        nheads: number of heads
        dim_feedforward: feature dimension in feedforward network
        enc_layers: number of Transformer encoder layers
        dec_layers: number of Transformer decoder layers
        pre_norm: whether to use pre-LayerNorm or not
        mask_dim: mask feature dimension
        enforce_input_project: add input project 1x1 conv even if input
            channels and hidden dim is identical
        d_model: transformer dimension
        dropout: dropout rate
        activation: activation function
        nhead: num heads in multi-head attention
        dec_n_points: number of sampling points in decoder
        return_intermediate_dec: return the intermediate results of decoder
        query_dim: 4 -> (x, y, w, h)
        dec_layer_share: whether to share each decoder layer
        semantic_ce_loss: use ce loss for semantic segmentation*/
    if (!two_stage || learn_tgt)
        query_feat = this->register_module("query_feat", torch::nn::Embedding(num_queries, hidden_dim));
    if (!two_stage && initialize_box_type.empty())
        query_embed = this->register_module("query_embed", torch::nn::Embedding(num_queries, 4));
    if (two_stage) {
        enc_output = this->register_module("enc_output", torch::nn::Linear(hidden_dim, hidden_dim));
        enc_output_norm = this->register_module("enc_output_norm", torch::nn::LayerNorm(hidden_dim));
    }
    input_proj = torch::nn::Sequential();
    for (int i = 0; i < num_feature_levels; ++i) {
        if (in_channels != hidden_dim || enforce_input_project) {
            torch::nn::Conv2d conv_layer = torch::nn::Conv2d(in_channels, hidden_dim, 1);
            c2_xavier_fill(conv_layer);
            input_proj->push_back(conv_layer);
        }
    }
    assert(mask_classification);
    if (mask_classification) {
        if (semantic_ce_loss)
            class_embed = this->register_module("class_embed", torch::nn::Linear(hidden_dim, num_classes+1));
        else
            class_embed = this->register_module("class_embed", torch::nn::Linear(hidden_dim, num_classes));
    }
    this->register_module("label_enc", torch::nn::Embedding(num_classes, hidden_dim));
    this->register_module("mask_embed", MLP(hidden_dim, hidden_dim, mask_dim, 3));

    this->register_module("decoder_norm", torch::nn::LayerNorm(hidden_dim));
    DeformableTransformerDecoderLayer decoder_layer = DeformableTransformerDecoderLayer(hidden_dim,
                                                                                        dim_feedforward,
                                                                                        dropout,
                                                                                        activation,
                                                                                        num_feature_levels,
                                                                                        nhead,
                                                                                        dec_n_points);
    TransformerDecoder decoder = TransformerDecoder(decoder_layer,
                                                    num_layers,
                                                    decoder_norm,
                                                    return_imediate_dec,
                                                    hidden_dim,
                                                    query_dim,
                                                    num_feature_levels,
                                                    dec_layer_share);
    _bbox_embed = MLPImpl(hidden_dim, hidden_dim, 4, 3);
    torch::nn::init::constant_(_bbox_embed.fc2->weight.data(), 0);
    torch::nn::init::constant_(_bbox_embed.fc2->bias.data(), 0);
    std::vector<MLPImpl> box_embed_layerlist{};
    for (int i = 0; i < num_layers; ++i)
        box_embed_layerlist.push_back(_bbox_embed);
    bbox_embed = torch::nn::Sequential(box_embed_layerlist);
    decoder.bbox_embed = bbox_embed;
}

std::vector<torch::Tensor> MaskDINO2Decoder::prepare_for_dn(std::vector< std::unordered_map<std::string, torch::Tensor> >& targets,
                                                            torch::Tensor& tgt,
                                                            torch::Tensor& refpoint_emb,
                                                            int batch_size)
{
    torch::Tensor input_query_label = torch::Tensor();
    torch::Tensor input_query_bbox = torch::Tensor();
    torch::Tensor attn_mask = torch::Tensor();
    std::unordered_map<std::string, torch::Tensor> mask_dict{};

    if (this->is_training()) {
        int scalar = this->dn_num;
        float noise_scale_local = this->noise_scale;

        std::vector<torch::Tensor> known, know_idx;
        std::vector<int> known_num;
        for (std::unordered_map<std::string, torch::Tensor>&t : targets)
            known.push_back(torch::ones_like(t["labels"]).cuda());
        for (torch::Tensor& k : known) {
            know_idx.push_back(torch::nonzero(k));
            known_num.push_back(sum(k).item<int>());
        }

        int max = *std::max_element(known_num.begin(), known_num.end());
        if (max > 0)
            scalar = (int)scalar / max;
        else
            scalar = 0;
        if (scalar == 0) {
            mask_dict_inclass = mask_dict; // saving locally as to return different objects     
            return std::vector<torch::Tensor> {input_query_label, input_query_bbox, attn_mask};
        }

        torch::Tensor unmask_bbox, unmask_label, labels, boxes, batch_idx;
        unmask_bbox = torch::cat(known);
        unmask_label = unmask_bbox;
        int idx = 0;
        for (std::unordered_map<std::string, torch::Tensor>&t : targets){
            torch::Tensor tmp_label = torch::cat(t["labels"]);
            labels = torch::cat({labels, tmp_label});
            torch::Tensor tmp_box = torch::cat(t["boxes"]);
            boxes = torch::cat({boxes, tmp_box});
            batch_idx = torch::cat(torch::full_like(t["labels"].to(torch::kLong), idx));
            ++idx;
        }
        torch::Tensor known_indice = torch::nonzero(unmask_label + unmask_bbox);
        known_indice = known_indice.view(-1);

        known_indice = known_indice.repeat(scalar).view(-1);
        torch::Tensor known_labels = labels.repeat(scalar).view(-1);
        torch::Tensor known_bid = batch_idx.repeat(scalar).view(-1);
        torch::Tensor known_bboxs = boxes.repeat(scalar);
        torch::Tensor known_labels_expaned = known_labels.clone();
        torch::Tensor known_bbox_expand = known_bboxs.clone();

        if (noise_scale > 0) {
            torch::Tensor p = torch::rand_like(known_labels_expaned.to(torch::kFloat));
            torch::Tensor chosen_indice = torch::nonzero(p < (noise_scale_local * 0.5)).view(-1);
            torch::Tensor new_label = torch::randint_like(chosen_indice, 0, num_classes);
            known_labels_expaned.scatter_(0, chosen_indice, new_label);
        }
        if (noise_scale > 0) {
            torch::Tensor diff = torch::zeros_like(known_bbox_expand);
            diff.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2)},
                             known_bbox_expand.index({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}) / 2);
            diff.index_put_({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)},
                             known_bbox_expand.index({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)}));
            known_bbox_expand += torch::mul((torch::rand_like(known_bbox_expand) * 2 - 1.0), diff.cuda() * noise_scale_local);
            known_bbox_expand = known_bbox_expand.clamp(0.0, 1.0);
        }

        torch::Tensor m = known_labels_expaned.to(torch::kLong).to("cuda");
        torch::Tensor input_label_embed = label_enc(m);
        torch::Tensor input_bbox_embed = inverse_sigmoid(known_bbox_expand);
        int single_pad = int(*std::max_element(known_num.begin(), known_num.end()));
        int pad_size = int(single_pad * scalar);

        torch::Tensor padding_label = torch::zeros({pad_size, hidden_dim}).cuda();
        torch::Tensor padding_bbox = torch::zeros({pad_size, 4}).cuda();

        torch::Tensor input_query_label, input_query_bbox;
        if (refpoint_emb.defined()) {
            input_query_label = torch::cat((padding_label, tgt), 0).repeat({batch_size, 1, 1});
            input_query_bbox = torch::cat((padding_bbox, refpoint_emb), 0).repeat({batch_size, 1, 1});
        }
        else {
            input_query_label = padding_label.repeat({batch_size, 1, 1});
            input_query_bbox = padding_bbox.repeat({batch_size, 1, 1});
        }

        torch::Tensor map_known_indice = torch::Tensor().to("cuda");
        if (known_num.empty()) {
            for (int& num : known_num)
                map_known_indice = torch::cat({map_known_indice, torch::linspace(0, num, num)});
            for (int i = 0; i < scalar; ++i)
                map_known_indice = torch::cat(map_known_indice + single_pad * i).to(torch::kLong);            
        }
        if (known_bid.defined()) {
            input_query_label.index_put_({known_bid.to(torch::kLong), map_known_indice}, input_label_embed);
            input_query_bbox.index_put_({known_bid.to(torch::kLong), map_known_indice}, input_bbox_embed);
        }
        
        int tgt_size = pad_size + this->num_queries;
        attn_mask = torch::ones({tgt_size, tgt_size}).to("cuda") < 0;
        attn_mask.index_put_({torch::indexing::Slice(pad_size, torch::indexing::None),
                              torch::indexing::Slice(torch::indexing::None, pad_size)}, true);
        
        for (int i = 0; i < scalar; ++i) {
            if (i == 0)
                attn_mask.index_put_({torch::indexing::Slice(single_pad * i, single_pad * (i+1)),
                                      torch::indexing::Slice(single_pad * (i+1), pad_size)},
                                      true);
            else if (i == scalar - 1)
                attn_mask.index_put_({torch::indexing::Slice(single_pad * i, single_pad * (i+1)),
                                      torch::indexing::Slice(torch::indexing::None, single_pad * i)},
                                      true);
            else {
                attn_mask.index_put_({torch::indexing::Slice(single_pad * i, single_pad * (i+1)),
                                      torch::indexing::Slice(single_pad * (i+1), pad_size)},
                                      true);
                attn_mask.index_put_({torch::indexing::Slice(single_pad * i, single_pad * (i+1)),
                                      torch::indexing::Slice(torch::indexing::None, single_pad * i)},
                                      true);
            }
        }
        mask_dict = {
            {"known_indice", known_indice.to(torch::kLong)},
            {"batch_idx", batch_idx.to(torch::kLong)},
            {"map_known_indice", map_known_indice.to(torch::kLong)},
            {"known_lbs_bboxes", (known_labels, known_bboxs)},
            {"know_idx", torch::from_blob(know_idx.data(), know_idx.size(), torch::TensorOptions().dtype(torch::kLong))},
            {"pad_size", torch::ones(1) * pad_size},
            {"scalar", torch::ones(1) * scalar}
        };
    }
    else {
        if (refpoint_emb.defined()) {
            input_query_label = tgt.repeat({batch_size, 1, 1});
            input_query_bbox = refpoint_emb.repeat({batch_size, 1, 1});
        }
    }
    mask_dict_inclass = mask_dict; // saving locally to return different objects from same function
    return std::vector<torch::Tensor> {input_query_label, input_query_bbox, attn_mask};
}

std::vector<torch::Tensor> MaskDINO2Decoder::dn_post_process(torch::Tensor& outputs_class,
                              torch::Tensor& outputs_coords,
                              std::unordered_map<std::string, torch::Tensor>& mask_dict,
                              torch::Tensor& outputs_mask)
{
    assert(mask_dict["pad_size"].item<int>() > 0);
    int pad_size = mask_dict["pad_size"].item<int>();
    torch::Tensor output_known_class = outputs_class.index({torch::indexing::Slice(),
                                                            torch::indexing::Slice(),
                                                            torch::indexing::Slice(torch::indexing::None, pad_size),
                                                            torch::indexing::Slice()});
    outputs_class = outputs_class.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice(pad_size, torch::indexing::None),
                                         torch::indexing::Slice()});
    torch::Tensor output_known_coord = outputs_coords.index({torch::indexing::Slice(),
                                                             torch::indexing::Slice(),
                                                             torch::indexing::Slice(torch::indexing::None, pad_size),
                                                             torch::indexing::Slice()});
    torch::Tensor outputs_coord = outputs_coords.index({torch::indexing::Slice(),
                                                        torch::indexing::Slice(),
                                                        torch::indexing::Slice(pad_size, torch::indexing::None),
                                                        torch::indexing::Slice()});
    mask_dict_inclass.insert({
        {"pred_logits", output_known_class.index({-1})},
        {"pred_boxes", output_known_coord.index({-1})}
    });
    
    torch::Tensor output_known_mask = torch::Tensor();
    if (outputs_mask.defined()) {
        output_known_mask = outputs_mask.index({torch::indexing::Slice(),
                                                torch::indexing::Slice(),
                                                torch::indexing::Slice(torch::indexing::None, pad_size),
                                                torch::indexing::Slice()});
        outputs_mask = outputs_mask.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(pad_size, torch::indexing::None),
                                           torch::indexing::Slice()});
        mask_dict_inclass.insert({"pred_masks", output_known_mask.index({-1})});
    }


    std::unordered_map<std::string, torch::Tensor> out_aux = _set_aux_loss(output_known_class, output_known_mask, output_known_coord);
    for (auto& [k, v] : out_aux) 
        mask_dict_inclass["aux_outputs_" + k] = v;
    

    return std::vector<torch::Tensor> {outputs_class, outputs_coord, outputs_mask};
}

torch::Tensor MaskDINO2Decoder::get_valid_ratio(torch::Tensor& mask) {
    int H = (int)mask.sizes()[1];
    int W = (int)mask.sizes()[2];
    torch::Tensor valid_H = torch::sum(~mask.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(),
                                                    0}),
                                       1);
    torch::Tensor valid_W = torch::sum(~mask.index({torch::indexing::Slice(),
                                                    0,
                                                    torch::indexing::Slice()}),
                                       1);
    torch::Tensor valid_ratio_h = valid_H.to(torch::kFloat) / H;
    torch::Tensor valid_ratio_w = valid_W.to(torch::kFloat) / W;
    torch::Tensor valid_ratio = torch::stack((valid_ratio_w, valid_ratio_h), -1);
    return valid_ratio;
}

torch::Tensor MaskDINO2Decoder::pred_box(std::vector<torch::Tensor>& reference,
                                         std::vector<torch::Tensor>& hs,
                                         torch::Tensor ref0 = torch::Tensor())
{
    torch::Device device = reference[0].device();
    
    std::vector<torch::Tensor> outputs_coord_list{};
    if (ref0.defined())
        outputs_coord_list.push_back(ref0.to(device));
    
    auto layer_ref_sig = reference.begin();
    auto layer_hs = hs.begin();
    int idx = 0;
    for (auto layer_bbox_embed = bbox_embed->begin(); layer_bbox_embed != bbox_embed->end(); ++layer_bbox_embed) {
        torch::Tensor layer_delta_unsig = layer_bbox_embed->forward(*layer_hs).to(device);
        torch::Tensor layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(*layer_ref_sig).to(device);
        layer_outputs_unsig = layer_outputs_unsig.sigmoid();
        ++idx;
    }
    torch::Tensor outputs_coord_tensor = torch::stack(outputs_coord_list);
    return outputs_coord_tensor;
}

std::unordered_map<std::string, torch::Tensor> MaskDINO2Decoder::forward(
    std::vector<torch::Tensor>& x,
    torch::Tensor& mask_features,
    std::vector<torch::Tensor>& masks,
    std::vector< std::unordered_map<std::string, torch::Tensor> >& targets)
{
    assert(int(x.size()) == num_feature_levels);
    torch::Device device = x[0].device();
    bool enable_mask = false;

    if (masks.size() > 0) {
        for (torch::Tensor& src : x) {
            if (src.sizes()[2] % 32 || src.sizes()[3] % 32)
                enable_mask = true;
        }
    }
    if (!enable_mask) {
        for (torch::Tensor& src : x)
            masks.push_back(torch::zeros({src.sizes()[0], src.sizes()[1], src.sizes()[2]},
                                         torch::TensorOptions().device(src.device()).dtype(torch::kBool)));
    }

    std::vector< std::pair<int, int> > size_list{}, spatial_shapes{};
    std::vector<torch::Tensor> src_flatten{}, mask_flatten{};
    int i = 0, bs;
    for (auto module = input_proj->begin(); module != input_proj->end(); ++module) {
        int idx = this->num_feature_levels - 1 - i;
        bs = (int)x[idx].sizes()[0];

        size_list.push_back(std::make_pair(x[i].sizes()[-2], x[i].sizes()[-1]));
        spatial_shapes.push_back(std::make_pair(x[idx].sizes()[-2], x[idx].sizes()[-1]));
        src_flatten.push_back(module->forward(x[idx]).flatten(2).transpose(1, 2));
        mask_flatten.push_back(masks[i].flatten(1));
        ++i;
    }
    torch::Tensor src_flatten_tensor = torch::cat(src_flatten, 1);
    torch::Tensor mask_flatten_tensor = torch::cat(mask_flatten, 1);
    torch::Tensor spatial_shapes_tensor = torch::from_blob(spatial_shapes.data(),
                                                           spatial_shapes.size(),
                                                           torch::TensorOptions().dtype(torch::kLong)
                                                                                 .device(src_flatten_tensor.device()));
    torch::Tensor level_start_index = torch::cat(
        (spatial_shapes_tensor.new_zeros({1}),
         spatial_shapes_tensor.prod(1)
                              .cumsum(0)
                              .index({torch::indexing::Slice(torch::indexing::None, -1)}))
    );
    std::vector<torch::Tensor> valid_ratios_vec;
    for (auto& m : masks)
        valid_ratios_vec.push_back(get_valid_ratio(m));
    torch::Tensor valid_ratios = torch::stack(valid_ratios_vec, 1);

    std::vector<torch::Tensor> predictions_class{}, predictions_mask{};
    torch::Tensor tgt, refpoint_embed;
    std::unordered_map<std::string, torch::Tensor> interm_outputs;
    if (this->two_stage) {
        std::vector<torch::Tensor> geop = gen_encoder_output_proposals(src_flatten_tensor, mask_flatten_tensor, spatial_shapes);
        torch::Tensor output_memory = geop[0], output_proposals = geop[1];
        output_memory = enc_output_norm(enc_output(output_memory));
        torch::Tensor enc_outputs_class_unselected = class_embed(output_memory);
        enc_outputs_class_unselected = this->_bbox_embed.forward(output_memory) + output_proposals;
        int topk = num_queries;
        torch::Tensor topk_proposals = std::get<1>(torch::topk(std::get<0>(enc_outputs_class_unselected.max(-1)), topk, 1));
        torch::Tensor refpoint_embed_undetach = torch::gather(enc_outputs_class_unselected, 1, topk_proposals.unsqueeze(-1).repeat({1, 1, 4}));
        refpoint_embed = refpoint_embed_undetach.detach();
        torch::Tensor tgt_undetach = torch::gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat({1, 1, hidden_dim})).transpose(0, 1);
        std::vector<torch::Tensor> fph = this->forward_prediction_heads(tgt_undetach, mask_features, this->is_training());
        torch::Tensor outputs_class = fph[0], outputs_mask = fph[1];

        tgt = tgt_undetach.detach();
        if (this->learn_tgt)
            tgt = this->query_feat->weight.index({torch::indexing::None}).repeat({bs, 1, 1});
        interm_outputs = {
            {"pred_logits", outputs_class},
            {"pred_boxes", refpoint_embed_undetach.sigmoid()},
            {"pred_masks", outputs_mask},
        };

        if (this->initialize_box_type != "no") {
            assert(this->initial_pred);
            torch::Tensor flatten_mask = outputs_mask.detach().flatten(0, 1);
            int h = (int)outputs_mask.sizes()[-2];
            int w = (int)outputs_mask.sizes()[-1];
            torch::Tensor bitmask_tensor = torch::greater(flatten_mask, 0);
            if (this->initialize_box_type != "bitmask") {//slow but more accurate
                torch::Tensor tensor = torch::greater(flatten_mask, 0);
                refpoint_embed = (BitMasks(bitmask_tensor).get_bounding_boxes()).tensor.to(device);
            }
            else if (this->initialize_box_type != "mask2box") //faster
                refpoint_embed = masks_to_boxes(bitmask_tensor);
            else
                std::cerr << "box type not implemented";
            std::vector<int> vec{w, h, w, h};
            refpoint_embed = box_xyxy_to_cxcywh(refpoint_embed)
                             / torch::from_blob(vec.data(), {4}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
            refpoint_embed = refpoint_embed.reshape({outputs_mask.sizes()[0], outputs_mask.sizes()[1], 4});
            refpoint_embed = inverse_sigmoid(refpoint_embed);
        }
    }
    else if (!this->two_stage) {
        tgt = this->query_feat->weight.index({torch::indexing::None}).repeat({bs, 1, 1});
        refpoint_embed = tgt;
    }
    torch::Tensor tgt_mask = torch::Tensor();
    // torch::Tensor mask_dict = torch::Tensor();
    if (this->dn != "no" && this->is_training()) {
        assert(!targets.empty());
        torch::Tensor empty_tensor1 = torch::Tensor(), empty_tensor2 = torch::Tensor();
        std::vector<torch::Tensor> pfd = this->prepare_for_dn(targets, empty_tensor1, empty_tensor2, (int)x[0].sizes()[0]);
        torch::Tensor input_query_label = pfd[0],
                      input_query_bbox = pfd[1],
                      tgt_mask = pfd[2],
                      mask_dict = pfd[3];
        if (mask_dict_inclass.size() > 0) {
            tgt = torch::cat({input_query_label, tgt}, 1);
            refpoint_embed = torch::cat({input_query_bbox, refpoint_embed}, 1);
        }
    }

    tgt = tgt.transpose(0, 1);
    if (this->initial_pred) {
        std::vector<torch::Tensor> outputs_vec = this->forward_prediction_heads(tgt, mask_features, this->is_training());
        torch::Tensor outputs_class = outputs_vec[0], outputs_mask = outputs_vec[1];
        predictions_class.push_back(outputs_class);
        predictions_mask.push_back(outputs_mask);
    }
    
    src_flatten_tensor = src_flatten_tensor.transpose(0, 1);

    std::vector< std::vector<torch::Tensor> > dec_out = decoder.forward(
        tgt,
        src_flatten_tensor,
        mask_flatten_tensor,
        torch::Tensor(),
        refpoint_embed.transpose(0, 1),
        level_start_index,
        spatial_shapes_tensor,
        valid_ratios,
        tgt_mask
    );
    std::vector<torch::Tensor> hs = dec_out[0], references = dec_out[1];

    i = 0;
    for (auto& output : hs) {
        torch::Tensor trans_out = output.transpose(0, -1);
        std::vector<torch::Tensor> fph_out = this->forward_prediction_heads(trans_out, mask_features, this->is_training() || hs.size() - 1);
        torch::Tensor outputs_class = fph_out[0], outputs_mask = fph_out[1];
        predictions_class.push_back(outputs_class);
        predictions_mask.push_back(outputs_mask);
    }

    torch::Tensor out_boxes;
    if (this->initial_pred) {
        out_boxes = this->pred_box(references, hs, refpoint_embed.sigmoid());
        assert((int)predictions_class.size() == num_layers + 1);
    }
    else
        out_boxes = this->pred_box(references, hs);
    
    torch::Tensor predictions_mask_tensor = torch::stack(predictions_mask);
    torch::Tensor predictions_class_tensor = torch::stack(predictions_class);
    if (mask_dict_inclass.size() > 0) {
        std::vector<torch::Tensor> dpp_out = this->dn_post_process(predictions_class_tensor, out_boxes, mask_dict_inclass, predictions_mask_tensor);
        predictions_class_tensor = dpp_out[0];
        out_boxes = dpp_out[1];
        predictions_mask_tensor = dpp_out[2];

        predictions_class_tensor.contiguous();
        std::vector<float> prediction_class_vec(
            predictions_class_tensor.data_ptr<float>(),
            predictions_class_tensor.data_ptr<float>() + predictions_class_tensor.numel()
        );

        predictions_mask_tensor.contiguous();
        std::vector<float> prediction_mask_vec(
            predictions_mask_tensor.data_ptr<float>(),
            predictions_mask_tensor.data_ptr<float>() + predictions_mask_tensor.numel()
        );
    }
    else if (this->is_training())
        predictions_class[-1] += 0.0*this->label_enc->weight.sum();
    
    if (this->mask_classification)
        predictions_class_tensor = torch::stack(predictions_class);
    else
        predictions_class_tensor = torch::Tensor();

    std::unordered_map<std::string, torch::Tensor> aux_outputs_map = _set_aux_loss(predictions_class_tensor,
                                                                                   predictions_mask_tensor,
                                                                                   out_boxes);

    std::unordered_map<std::string, torch::Tensor> out = {
        {"pred_logits", predictions_class[-1]},
        {"pred_boxes", predictions_mask[-1]},
        {"pred_masks", out_boxes[-1]}
    };
    for (auto& [k, v] : aux_outputs_map)
        out["aux_outputs_" + k] = v;

    if (this->two_stage) {
        for (auto& [k, v] : interm_outputs)
            out["interm_ouputs_" + k] = v;
    }
    for (auto& [k, v] : mask_dict_inclass)
        out["mask_dict_" + k] = v;

    return out;
}

std::vector<torch::Tensor> MaskDINO2Decoder::forward_prediction_heads(
    torch::Tensor& output,
    torch::Tensor& mask_features,
    bool pred_mask = true)
{
    torch::Tensor decoder_output = this->decoder_norm(output);
    decoder_output = decoder_output.transpose(0, 1);
    torch::Tensor outputs_class = this->class_embed(decoder_output);
    torch::Tensor outputs_mask{};
    if (pred_mask) {
        torch::Tensor mask_embed_tensor = mask_embed.forward(decoder_output);
        outputs_mask = torch::einsum("bqc,bchw->bqhw", {mask_embed_tensor, mask_features});
    }

    return std::vector<torch::Tensor> {outputs_class, outputs_mask};
}

std::unordered_map<std::string, torch::Tensor> MaskDINO2Decoder::_set_aux_loss(
    torch::Tensor& outputs_class,
    torch::Tensor& outputs_seg_masks,
    torch::Tensor out_boxes = torch::Tensor())
{
    std::unordered_map<std::string, torch::Tensor> ret;
    torch::Tensor oc = outputs_class.index({torch::indexing::Slice(torch::indexing::None, -1)});
    torch::Tensor osm = outputs_seg_masks.index({torch::indexing::Slice(torch::indexing::None, -1)});
    torch::Tensor *oc_ptr = (torch::Tensor*)oc.data_ptr();
    torch::Tensor *osm_ptr = (torch::Tensor*)osm.data_ptr();
    if (out_boxes.defined()) {
        std::unordered_map<std::string, torch::Tensor> ret;
        torch::Tensor oc = outputs_class.index({torch::indexing::Slice(torch::indexing::None, -1)});
        torch::Tensor osm = outputs_seg_masks.index({torch::indexing::Slice(torch::indexing::None, -1)});
        torch::Tensor *oc_ptr = (torch::Tensor*)oc.data_ptr();
        torch::Tensor *osm_ptr = (torch::Tensor*)osm.data_ptr();
        for (int i = 0; i < oc.sizes()[0]; ++i)
            ret.insert({{"pred_logits", *oc_ptr++},
                        {"pred_masks", *osm_ptr++}});
        return ret;
    }
    else {
        torch::Tensor ob = out_boxes.index({torch::indexing::Slice(torch::indexing::None, -1)});
        torch::Tensor *ob_ptr = (torch::Tensor*)ob.data_ptr();
        for (int i = 0; i < oc.sizes()[0]; ++i)
            ret.insert({{"pred_logits", *oc_ptr++},
                        {"pred_masks", *osm_ptr++},
                        {"pred_boxes", *ob_ptr++}});
        return ret;
    }
}
