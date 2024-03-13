#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <torch/torch.h>
#include <valarray>
#include <vector>
#include "backbone.h"
#include "detectron2.h"
#include "matcher.h"
#include "memory.h"
#include "utils/box_operations.h"
#include "utils/modules.h"


struct SwinTransformerBlockImpl : torch::nn::Module {
/*
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

*/
public:
    SwinTransformerBlockImpl(int, int, int, int, float, float,
                             float, float, float, bool);
    // class methods
    torch::Tensor forward(torch::Tensor, torch::Tensor);
    void set_H(int h) { H = h; }
    void set_W(int w) { W = w; }

private:
    int
        dim,
        num_heads,
        window_size,
        shift_size,
        H,
        W;
    float
        mlp_ratio,
            qk_scale,
            drop,
            attn_drop,
            drop_path_ratio;
    bool
        qkv_bias;
    DropPath
        drop_path;
    torch::nn::Identity
        non_drop_path = nullptr;
    torch::nn::GELU
        act_layer = torch::nn::GELU();
    torch::nn::LayerNorm
        norm1 = nullptr,
        norm2 = nullptr;
    WindowAttention
        attn;
    MLPImpl
        mlp;
};
TORCH_MODULE(SwinTransformerBlock);

SwinTransformerBlockImpl::SwinTransformerBlockImpl(
    int d,
    int nh,
    int ws = 7,
    int ss = 0,
    float mr = 4.0,
    float qks = -1.,
    float dr = 0.0,
    float ad = 0.0,
    float dp = 0.0,
    bool qkvb = true
) : dim(d),
    num_heads(nh),
    window_size(ws),
    shift_size(ss),
    mlp_ratio(mr),
    qk_scale(qks),
    drop(dr),
    attn_drop(ad),
    drop_path_ratio(dp),
    qkv_bias(qkvb)
{
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    attn = WindowAttention(dim,
                            std::vector<int> {window_size},
                            num_heads,
                            qkv_bias,
                            qk_scale,
                            attn_drop,
                            drop);
    if (drop_path_ratio > 0.0) {
        drop_path = DropPath(dp);
    }
    else {
        non_drop_path = register_module("drop_path", torch::nn::Identity());
    }
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    int mlp_hidden_dim = int(dim * mlp_ratio);
    mlp = MLPImpl(dim, mlp_hidden_dim, dim, drop);

}

torch::Tensor SwinTransformerBlockImpl::forward(torch::Tensor x, torch::Tensor mask_matrix) {
/*
    Forward function.
    Args:
        x: Input feature, tensor size (B, H*W, C).
        H, W: Spatial resolution of the input feature.
        mask_matrix: Attention mask for cyclic shift.
*/
    int B = int(x.sizes()[0]);
    int L = int(x.sizes()[1]);
    int C = int(x.sizes()[2]);
    if (H != 0 && W != 0 && L == H * W) {
        std::cerr << "input feature has wrong size\n";
    }
    torch::Tensor shortcut = x;
    x = norm1(x);
    x = x.view({B, H, W, C});
    int pad_l = 0, pad_t = 0;
    int pad_r = (window_size - W % window_size) % window_size;
    int pad_b = (window_size - H % window_size) % window_size;
    x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 0, pad_l, pad_r, pad_t, pad_b}));
    int Hp = int(x.sizes()[1]);
    int Wp = int(x.sizes()[2]);

    torch::Tensor shifted_x;
    torch::Tensor attn_mask;
    if (shift_size > 0) {
        torch::Tensor shifted_x = torch::roll(x, (-shift_size, -shift_size), (1, 2));
        torch::Tensor attn_mask = mask_matrix;
    }
    else {
        torch::Tensor shifted_x = x;
    }
    
    torch::Tensor x_windows = window_partition(shifted_x, window_size);
    x_windows = x_windows.view({-1, window_size*window_size, C});

    torch::Tensor attn_windows = attn.forward(x_windows, attn_mask);
    attn_windows = attn_windows.view({-1, window_size, window_size, C});
    shifted_x = window_reverse(attn_windows, window_size, Hp, Wp);

    if (shift_size > 0) {
        x = torch::roll(shifted_x, (shift_size, shift_size), (1, 2));
    }
    else {
        x = shifted_x;
    }
    if (pad_r > 0 || pad_b > 0) {
        x = x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(torch::indexing::None, H),
                     torch::indexing::Slice(torch::indexing::None, W),
                     torch::indexing::Slice()})
                     .contiguous();
    }
    x = x.view({B, H * W, C});
    x = shortcut + drop_path.drop_path(x);
    x = norm2(x);
    x = x + drop_path.drop_path(mlp.forward(x));

    return x;
}


class PatchMerging : torch::nn::Module {
/*
Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
*/
public:
    PatchMerging(int d) :
        dim(d),
        reduction(torch::nn::LinearOptions(4*d, 2*d).bias(true)),
        norm_layer(torch::nn::LayerNorm(torch::nn::LayerNormOptions({int(4*d)})))
        {  }
    
    torch::Tensor forward(torch::Tensor&, int&, int&);

private:
    int dim;
    torch::nn::Linear reduction = nullptr;
    torch::nn::LayerNorm norm_layer = nullptr;
};

torch::Tensor PatchMerging::forward(torch::Tensor &x, int &H, int &W) {
    int B = int(x.sizes()[0]);
    int L = int(x.sizes()[1]);
    int C = int(x.sizes()[2]);
    if (H != 0 && W != 0 && L == H * W)
        std::cerr << "input feature has wrong size\n";
    
    x = x.view({B, H, W, C});
    bool pad_input = (H % 2 == 1) || (W % 2 == 1);
    if (pad_input)
        x = torch::nn::functional::pad(x,
            torch::nn::functional::PadFuncOptions({0, 0, 0, W % 2, 0, H % 2}));
    torch::Tensor x0 = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(0, torch::indexing::None, 2),
                                torch::indexing::Slice(0, torch::indexing::None, 2),
                                torch::indexing::Slice()});
    torch::Tensor x1 = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(1, torch::indexing::None, 2),
                                torch::indexing::Slice(0, torch::indexing::None, 2),
                                torch::indexing::Slice()});
    torch::Tensor x2 = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(0, torch::indexing::None, 2),
                                torch::indexing::Slice(1, torch::indexing::None, 2),
                                torch::indexing::Slice()});
    torch::Tensor x3 = x.index({torch::indexing::Slice(),
                                torch::indexing::Slice(1, torch::indexing::None, 2),
                                torch::indexing::Slice(1, torch::indexing::None, 2),
                                torch::indexing::Slice()});
    x = torch::cat({x0, x1, x2, x3}, -1);
    x = x.view({B, -1, 4 * C});
    x = norm_layer(x);
    x = reduction(x);
    return x;
}

struct BasicLayerImpl : torch::nn::Module {
/*
    A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
*/
public:
    BasicLayerImpl(int, int, int, int, float, float,
                   float, float, float, bool, bool);

    std::vector<torch::Tensor> forward(torch::Tensor, int, int);

private:
    int
        depth,
        window_size,
        shift_size;
    bool
        use_checkpoint;
};
TORCH_MODULE(BasicLayer);

BasicLayerImpl::BasicLayerImpl(
    int dim,
    int dep,
    int num_heads,
    int ws = 7,
    float mr = 4.0,
    float qks = -1.,
    float dp = 0.0,
    float ad = 0.0,
    float dpr = 0.0,
    bool qkvb = true,
    bool uc = false
) : window_size(ws),
    shift_size(int(ws/2)),
    depth(dep),
    use_checkpoint(uc)
{
    for (int i = 0; i < depth; ++i) {
        register_module(
            "SwinTransformerBlock" + i,
            SwinTransformerBlock(
                dim,
                num_heads,
                ws,
                shift_size,
                mr,
                qks,
                dpr,
                ad,
                dp,
                qkvb
            ));
    }
    // left out downsample layer; not needed for purposes of this project
}

std::vector<torch::Tensor> BasicLayerImpl::forward(torch::Tensor x, int H, int W) {
/*
    Forward function.
    Args:
        x: Input feature, tensor size (B, H*W, C).
        H, W: Spatial resolution of the input feature.
*/
    int Hp = int(ceil(H / window_size)) * window_size;
    int Wp = int(ceil(W / window_size)) * window_size;
    torch::Tensor img_mask = torch::zeros({1, Hp, Wp, 1}, x.device());
    torch::indexing::Slice h_slices[3] = {torch::indexing::Slice(0, -window_size),
                              torch::indexing::Slice(-window_size, -shift_size),
                              torch::indexing::Slice(-shift_size, 0)};
    torch::indexing::Slice w_slices[3] = {torch::indexing::Slice(0, -window_size),
                              torch::indexing::Slice(-window_size, -shift_size),
                              torch::indexing::Slice(-shift_size, 0)};
    int cnt = 0;
    for (torch::indexing::Slice h : h_slices) {
        for (torch::indexing::Slice w : w_slices) {
            img_mask.index_put_({torch::indexing::Slice(),
                                h,
                                w,
                                torch::indexing::Slice()},
                                cnt);
            ++cnt;            
        }
    }
    torch::Tensor mask_windows = window_partition(img_mask, window_size);
    mask_windows = mask_windows.view({-1, window_size*window_size});
    torch::Tensor attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2);
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0));
    
    for (auto& mod : this->named_children()) {
        SwinTransformerBlockImpl block = *(mod.value())->as<SwinTransformerBlock>();
        block.set_H(H);
        block.set_W(W);
        x = block.forward(x, attn_mask);
    }

    return std::vector<torch::Tensor>{x, torch::tensor({H, W})};
}


struct PatchEmbed : torch::nn::Module {
/*
Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
*/
public:
    PatchEmbed(int ps = 4, int ic = 3, int ed= 96, bool has_norm = false) : 
            patch_size(ps), in_chan(ic), embed_dim(ed), has_norm(has_norm)
    { 
        proj = register_module("proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_chan, embed_dim, patch_size).stride(patch_size)));
        if (has_norm) {
            norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ed})));
        }
    }
    torch::Tensor forward(torch::Tensor);

private:
    int
        patch_size,
        in_chan,
        embed_dim;
    bool
        has_norm;
    torch::nn::Conv2d
        proj = nullptr;
    torch::nn::LayerNorm
        norm = nullptr;
};

torch::Tensor PatchEmbed::forward(torch::Tensor x) {
    int H = int(x.sizes()[2]);
    int W = int(x.sizes()[3]);
    if (W % patch_size != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, patch_size - W % patch_size}));
    }
    if (H % patch_size != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, patch_size - H % patch_size}));
    }
    x = proj(x);
    if (has_norm) {
        int Wh = int(x.sizes()[2]);
        int Ww = int(x.sizes()[3]);
        x = x.flatten(2).transpose(1, 2);
        x = norm(x);
        x = x.transpose(1, 2).view({-1, embed_dim, Wh, Ww});
    }
    return x;
}


class SwinTransformer : torch::nn::Module {
/*
    Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
*/
public:
    SwinTransformer(int, int, int, int, std::vector<int>, std::vector<int>,
                    int, float, bool, float, float, float, float,
                    bool, bool, std::vector<int>, int, bool);
    
    void freeze_stages();
    void init_weights(std::string &pretrained);
    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor x);
    void Train(bool mode = true) {
        this->train(mode);
        freeze_stages();
    }

    int
        pretrain_img_size,
        num_layers,
        embed_dim,
        frozen_stages;
    bool
        ape,
        patch_norm;
    float
        drop_rate;
    std::vector<int>
        out_ind,
        num_heads,
        num_features;
    PatchEmbed
        patch_embed;
    torch::nn::Dropout
        pos_drop = nullptr;
};

SwinTransformer::SwinTransformer(
    int pis = 224,
    int patch_size = 4,
    int in_chans = 3,
    int ed = 96,
    std::vector<int> depths = {2, 2, 6, 2},
    std::vector<int> num_heads = {3, 6, 12, 24},
    int window_size = 7,
    float mlp_ratio = 4.0,
    bool qkv_bias = true,
    float qk_scale = -1,
    float drop_rate = 0.0,
    float attn_drop_rate = 0.0, 
    float drop_path_rate = 0.2,
    bool ape = false,
    bool patch_norm = true,
    std::vector<int> out_indices = {0, 1, 2, 3},
    int frozen_stages = -1,
    bool use_checkpoint = false
) : pretrain_img_size(pis),
    num_layers(int(depths.size())), 
    embed_dim(ed), ape(ape),
    patch_norm(patch_norm),
    out_ind(out_indices),
    frozen_stages(frozen_stages),
    drop_rate(drop_rate)
{
    patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, patch_norm);
    if (ape) {
        int patches_resolution = pretrain_img_size / patch_size;
        torch::Tensor &t = register_parameter("absolute_pos_embed",
                                              torch::zeros({1, embed_dim, patches_resolution, patches_resolution}));
        double l = (1. + erf((-2/0.02) / sqrt(2.))) / 2.;
        double u = (1. + erf((2/0.02) / sqrt(2.))) / 2.;
        t.uniform_(2 * l - 1, 2 * u - 1);
        t.erfinv_();
        t.mul_(0.02 * sqrt(2.));
        t.add_(0);
        t.clamp_(-2., 2.);
    }
    pos_drop = torch::nn::Dropout(drop_rate);
    int depths_sum = 0;
    for (auto& item : depths)
        depths_sum += item;
    torch::Tensor dpr = torch::linspace(0, drop_path_rate, depths_sum);

    for (int i_layer = 0; i_layer < num_layers; ++i_layer) {
        BasicLayer layer = BasicLayer(
            int(embed_dim * pow(2, i_layer)),
            depths[i_layer],
            num_heads[i_layer],
            window_size,
            mlp_ratio,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            qkv_bias,
            use_checkpoint
        );
        register_module<BasicLayerImpl>("BasicLayer" + i_layer, layer);
    }
    
    for (int i = 0; i < num_layers; ++i)
        num_features.push_back(int(embed_dim * pow(2, i)));
    
    for (auto& i_layer : out_ind) {
        torch::nn::LayerNorm layer = torch::nn::LayerNorm(torch::nn::LayerNormOptions({num_features[i_layer]}));
        register_module("norm" + i_layer, layer);
    }
    freeze_stages();
}

void SwinTransformer::freeze_stages() {
    if (frozen_stages >= 0) {
        patch_embed.eval();
        for (auto& param : patch_embed.parameters()) {
            param.requires_grad_(false);
        }
    }
    else if (frozen_stages >= 1 && ape) {
        this->named_parameters()["absolute_pos_embed"].requires_grad_(false);
    }
    else if (frozen_stages >= 2) {
        pos_drop = torch::nn::Dropout(torch::nn::DropoutOptions().p(0.0));
        for (int i = 0; i < (frozen_stages - 1); ++i) {
            auto &layer = named_modules()["BasicLayer" + i];
        }
    }
}

std::unordered_map<std::string, torch::Tensor> SwinTransformer::forward(torch::Tensor x) {
    x = patch_embed.forward(x);
    int64_t Wh = x.sizes()[2];
    int64_t Ww = x.sizes()[3];
    if (ape) {
        torch::Tensor absolute_pos_embed = torch::nn::functional::interpolate(
            this->named_parameters()["absolute_pos_embed"],
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{Wh, Ww}).mode(torch::kBicubic)
        );
        x = (x + absolute_pos_embed).flatten(2).transpose(1, 2);
    }
    else {
        x = x.flatten(2).transpose(1, 2);
    }
    x = pos_drop(x);

    std::unordered_map<std::string, torch::Tensor> outs{};
    for (int i = 0; i < num_layers; ++i) {
        auto &layer = named_modules()["BasicLayer" + i];
        std::vector<torch::Tensor> res = layer->as<BasicLayer>()->forward(x, Wh, Ww);
        x = res[0];
        Wh = *res[1].select(0,0).data_ptr<int64_t>();
        Ww = *res[1].select(0,1).data_ptr<int64_t>();
        
        if (std::find(out_ind.begin(), out_ind.end(), i) != out_ind.end()) {
            auto &norm_layer = named_modules()["norm" + i];
            x = norm_layer->as<torch::nn::LayerNorm>()->forward(x);
            torch::Tensor out = x.view({-1, Wh, Ww, num_features[i]}).permute({0, 3, 1, 2}).contiguous();
            outs.insert({"res" + i + 2, out});
        }
    }
    return outs;
}


class D2SwinTransformer : Backbone {
public:
    D2SwinTransformer(std::vector<std::string>, int, int, int, int,
                    std::vector<int>, std::vector<int>, int, float, bool,
                    float, float, float, float,
                    bool, bool, std::vector<int>, int, bool);
    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor);
    std::unordered_map<std::string, ShapeSpec> output_shape();
    inline int size_divisibility() const { return 32; }

private:
    std::vector<std::string>
        out_features;
    SwinTransformer
        swintransformer;
    std::unordered_map<std::string, int>
        out_feature_strides{},
        out_feature_channels{};
};

D2SwinTransformer::D2SwinTransformer(
    std::vector<std::string> out_features,
    int pis = 224,
    int patch_size = 4,
    int in_chans = 3,
    int ed = 96,
    std::vector<int> depths = {2, 2, 6, 2},
    std::vector<int> num_heads = {3, 6, 12, 24},
    int window_size = 7,
    float mlp_ratio = 4.0,
    bool qkv_bias = true,
    float qk_scale = -1,
    float drop_rate = 0.0,
    float attn_drop_rate = 0.0, 
    float drop_path_rate = 0.2,
    bool ape = false,
    bool patch_norm = true,
    std::vector<int> out_indices = {0, 1, 2, 3},
    int frozen_stages = -1,
    bool use_checkpoint = false
) : out_features(out_features)
{
    swintransformer = SwinTransformer(pis, patch_size, in_chans, ed, depths, num_heads, window_size,
                                      mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                      ape, patch_norm, out_indices, frozen_stages, use_checkpoint);
    out_feature_strides.insert({{"res2", 4}, {"res3", 8}, {"res4", 16}, {"res5", 32}});
    out_feature_channels.insert({{"res2", swintransformer.num_features[0]},
                                 {"res3", swintransformer.num_features[1]},
                                 {"res4", swintransformer.num_features[2]},
                                 {"res5", swintransformer.num_features[3]}});
}

std::unordered_map<std::string, torch::Tensor> D2SwinTransformer::forward(torch::Tensor x) {
/*
    Args:
        x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
    Returns:
        dict[str->Tensor]: names and the corresponding features
*/
    if (x.dim() != 4) {
        std::cerr << "SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!";
    }
    std::unordered_map<std::string, torch::Tensor> outputs;
    std::unordered_map<std::string, torch::Tensor> y = swintransformer.forward(x);
    for (const auto & [ k, v ] : y) {
        if (std::find(out_features.begin(), out_features.end(), k) != out_features.end()) {
            outputs.insert({k, v});
        }
    }
    return outputs;
}

std::unordered_map<std::string, ShapeSpec> D2SwinTransformer::output_shape() {
    std::unordered_map<std::string, ShapeSpec> ret = {};
    for (auto &name : out_features) {
        ret.insert({name, ShapeSpec(out_feature_channels[name], out_feature_strides[name])});
    }
    return ret;
}
