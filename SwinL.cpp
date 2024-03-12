#include <array>
#include <cmath>
#include <cstddef>
#include <valarray>
#include <iostream>
#include <torch/torch.h>
#include <vector>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "include/structures/image_list.h"
#include "memory.h"


class MLPImpl : torch::nn::Module {
public:
    MLPImpl() { } // default constructor for SwinTransformerBlock declaration
    MLPImpl(int in_features, int hidden_features, int out_features, float d = 0.0) : 
        fc1(in_features, hidden_features), fc2(hidden_features, out_features), drop(d) { }
    torch::Tensor forward(torch::Tensor &);

    torch::nn::Linear fc1 = nullptr, fc2 = nullptr;
    torch::nn::Dropout drop = nullptr;
    torch::nn::GELU act = torch::nn::GELU();
};
TORCH_MODULE(MLP);

torch::Tensor MLPImpl::forward(torch::Tensor& x) {
    x = fc1(x);
    x = act(x);
    x = drop(x);
    x = fc2(x);
    x = drop(x);
    return x;
}

torch::Tensor window_partition(torch::Tensor x, int window_size) {
    int B = int(x.sizes()[0]);
    int H = int(x.sizes()[1]);
    int W = int(x.sizes()[2]);
    int C = int(x.sizes()[3]);
    x = x.view({B, int(H / window_size), window_size, int(W / window_size), window_size, C});
    torch::Tensor windows = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({-1, window_size, window_size, C});
    return windows;
}

torch::Tensor window_reverse(torch::Tensor windows, int window_size, int H, int W) {
    int B = int(windows.sizes()[0] / (H * W / window_size / window_size));
    torch::Tensor x = windows.view({B, int(H / window_size), int(W / window_size), window_size, window_size, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({B, H, W, -1});
    return x;
}


class WindowAttention : torch::nn::Module {
public:
    WindowAttention() { }; // default constructor to create SwinTransformerBlock constructor
    WindowAttention(const int d, std::vector<int> ws, const int nh,
                    const bool qkvb = true, const double qks = -1,
                    const float ad = 0.0, const float pd = 0.0) : dim(d),
                    window_size(ws), num_heads(nh), qkv_bias(qkvb),
                    qk_scale(qks), attn_drop_ratio(ad), proj_drop_ratio(pd)
    {
        this->register_parameter("relative_position_bias_table",
            torch::zeros({(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads}));
        if (qks == -1) {
            qk_scale = pow(int(dim / num_heads), -0.5);
        }
        qkv = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3).bias(qkv_bias)));
        attn_drop = register_module("attn_drop", torch::nn::Dropout(attn_drop_ratio));
        proj = register_module("proj", torch::nn::Linear(dim, dim));
        proj_drop = register_module("proj_drop", torch::nn::Dropout(proj_drop_ratio));
        softmax = register_module("softmax", torch::nn::Softmax(-1));
        assemble();
    };

    void assemble();

    torch::Tensor forward(torch::Tensor, torch::Tensor);

private:
    int dim;
    std::vector<int> window_size;
    int num_heads;
    bool qkv_bias;
    double qk_scale;
    float attn_drop_ratio;
    float proj_drop_ratio;
    torch::nn::Linear qkv = nullptr, proj = nullptr;
    torch::nn::Dropout attn_drop = nullptr, proj_drop = nullptr;
    torch::nn::Softmax softmax = nullptr;
};

void WindowAttention::assemble() {
    torch::Tensor coords_h = torch::arange(window_size[0]);
    torch::Tensor coords_w = torch::arange(window_size[1]);
    torch::Tensor coords = torch::stack(torch::meshgrid({coords_h, coords_w}));
    torch::Tensor coords_flatten = torch::flatten(coords, 1);
    torch::Tensor relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1);
    relative_coords = relative_coords.permute({1, 2, 0}).contiguous();
    torch::Tensor tmp = torch::zeros(relative_coords.sizes());
    tmp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, window_size[0] - 1);
    tmp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1}, window_size[1] - 1);
    relative_coords += tmp;
    // torch::Tensor tmp = torch::zeros(relative_coords.sizes());
    tmp.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, 2*window_size[1] - 1);
    relative_coords *= tmp;
    torch::Tensor relative_position_index = relative_coords.sum(-1);
    this->register_buffer("relative_position_index", relative_position_index);

    torch::Tensor &t = this->named_parameters()["relative_position_bias_table"];
    double l = (1. + erf((-2/0.02) / sqrt(2.))) / 2.;
    double u = (1. + erf((2/0.02) / sqrt(2.))) / 2.;
    t.uniform_(2 * l - 1, 2 * u - 1);
    t.erfinv_();
    t.mul_(0.02 * sqrt(2.));
    t.add_(0);
    t.clamp_(-2., 2.);
}

torch::Tensor WindowAttention::forward(torch::Tensor x, torch::Tensor mask = torch::ones({1})) {
    int B_ = int(x.sizes()[0]);
    int N = int(x.sizes()[1]);
    int C = int(x.sizes()[2]);
    torch::Tensor qkv_tensor = (
        qkv(x)
        .reshape({B_, N, 3, num_heads, int(C / num_heads)})
        .permute({2, 0, 3, 1, 4})
    );
    torch::Tensor q = qkv_tensor.index({0});
    torch::Tensor k = qkv_tensor.index({1});
    torch::Tensor v = qkv_tensor.index({2});
    q *= qk_scale;
    torch::Tensor attn = torch::mm(q, k.transpose(-2, -1));
    torch::Tensor &rpbt = this->named_parameters()["relative_position_bias_table"];
    torch::Tensor &rpi = this->named_parameters()["relative_position_index"];
    torch::Tensor relative_position_bias = rpbt[rpi.view(-1)].view({window_size[0]*window_size[1], window_size[0]*window_size[1], -1});
    relative_position_bias = relative_position_bias.permute({2, 0, 1}).contiguous();
    attn = attn + relative_position_bias.unsqueeze(0);
    if ( !mask.equal(torch::ones({1})) ) {
        int nW = int(mask.sizes()[0]);
        attn = attn.view({int(B_ / nW), nW, num_heads, N, N}) + mask.unsqueeze(1).unsqueeze(0);
        attn = attn.view({-1, num_heads, N, N});
    }
    attn = softmax(attn);
    attn = attn_drop(attn);

    x = (torch::mm(attn, v)).transpose(1, 2).reshape({B_, N, C});
    x = proj(x);
    x = proj_drop(x);
    return x;
}


class DropPath : torch::nn::Module {
public:
    DropPath(float dp = 0., bool sbk = true, bool t = false) : drop_prob(dp), scale_by_keep(sbk), training(t) { }
    torch::Tensor drop_path(torch::Tensor, float, bool, bool);
    torch::Tensor forward(torch::Tensor x) { return drop_path(x, drop_prob, training, scale_by_keep); }

// private:
    float drop_prob;
    bool scale_by_keep, training;
};

torch::Tensor DropPath::drop_path(torch::Tensor x, float drop_prob = 0., bool training = false, bool scale_by_keep = true) {
    if (drop_prob == 0. || training) {
        return x;
    }
    float keep_prob = 1 - drop_prob;
    auto shape = (x.sizes()[0]) + (1) * (x.dim() - 1);
    torch::Tensor random_tensor = x.new_empty(shape).bernoulli_(keep_prob);
    if (keep_prob > 0.0 && scale_by_keep) {
        random_tensor.div_(keep_prob);
    }
    return x * random_tensor;
}


struct SwinTransformerBlockImpl : torch::nn::Module {
public:
    SwinTransformerBlockImpl(int, int, int, int, float, float,
                             float, float, float, bool);
    torch::Tensor forward(torch::Tensor, torch::Tensor);
    void set_H(int h) { H = h; }
    void set_W(int w) { W = w; }

private:
    int dim, num_heads, window_size, shift_size, H, W;
    float mlp_ratio, qk_scale, drop, attn_drop, drop_path_ratio;
    bool qkv_bias;
    DropPath drop_path;
    torch::nn::Identity non_drop_path = nullptr;
    torch::nn::GELU act_layer = torch::nn::GELU();
    torch::nn::LayerNorm norm1 = nullptr, norm2 = nullptr;
    WindowAttention attn;
    MLP mlp;
};
TORCH_MODULE(SwinTransformerBlock);

SwinTransformerBlockImpl::SwinTransformerBlockImpl(int d, int nh, int ws = 7, int ss = 0, float mr = 4.0, float qks = -1.,
                                           float dr = 0.0, float ad = 0.0, float dp = 0.0, bool qkvb = true) :
                                           dim(d), num_heads(nh), window_size(ws), shift_size(ss), mlp_ratio(mr),
                                           qk_scale(qks), drop(dr), attn_drop(ad), drop_path_ratio(dp), qkv_bias(qkvb)
{
    norm1 = register_module("norm1", torch::nn::LayerNorm(dim));
    attn = WindowAttention(dim,
                            std::vector<int> {window_size},
                            num_heads,
                            qkv_bias,
                            qk_scale,
                            attn_drop,
                            drop);
    if (drop_path_ratio > 0.0)
        drop_path = DropPath(dp);
    else
        non_drop_path = register_module("drop_path", torch::nn::Identity());
    norm2 = register_module("norm2", torch::nn::LayerNorm(dim));
    int mlp_hidden_dim = int(dim * mlp_ratio);
    mlp = MLP(dim, mlp_hidden_dim, dim, drop);

}

torch::Tensor SwinTransformerBlockImpl::forward(torch::Tensor x, torch::Tensor mask_matrix) {
    int B = int(x.sizes()[0]);
    int L = int(x.sizes()[1]);
    int C = int(x.sizes()[2]);
    if (H != 0 && W != 0 && L == H * W)
        std::cerr << "input feature has wrong size\n";
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

    if (shift_size > 0)
        x = torch::roll(shifted_x, (shift_size, shift_size), (1, 2));
    else
        x = shifted_x;
    
    if (pad_r > 0 || pad_b > 0)
        x = x.index({torch::indexing::Slice(),
                     torch::indexing::Slice(torch::indexing::None, H),
                     torch::indexing::Slice(torch::indexing::None, W),
                     torch::indexing::Slice()})
                     .contiguous();
    
    x = x.view({B, H * W, C});

    x = shortcut + drop_path.drop_path(x);
    x = norm2(x);
    x = x + drop_path.drop_path(mlp->forward(x));

    return x;
}


class PatchMerging : torch::nn::Module {
public:
    PatchMerging(int d) : dim(d), reduction(4*d, 2*d, true), //took out norm_layer parameter bc it seemed pointless
                          norm_layer(torch::nn::LayerNorm(4*d))  {  }
    
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
public:
    BasicLayerImpl(int, int, int, int, float, float,
               float, float, float, bool, bool);

    std::vector<torch::Tensor> forward(torch::Tensor, int, int);

private:
    int depth, window_size, shift_size;
    bool use_checkpoint;
};
TORCH_MODULE(BasicLayer);

BasicLayerImpl::BasicLayerImpl(int dim, int dep, int num_heads, int ws = 7, float mr = 4.0,
                       float qks = -1., float dp = 0.0, float ad = 0.0, float dpr = 0.0,
                       bool qkvb = true, bool uc = false) : window_size(ws),
                       shift_size(int(ws/2)), depth(dep), use_checkpoint(uc)
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
    
    // fix assigning each module H and W attributes, downsampling option, and returning H and W
    for (auto& mod : this->named_children()) {
        SwinTransformerBlock block = static_cast<SwinTransformerBlock>(mod.value());
        block->set_H(H);
        block->set_W(W);
        x = block->forward(x, attn_mask);
    }

    return std::vector<torch::Tensor>{x, torch::tensor({H, W})};
}


struct PatchEmbed : torch::nn::Module {
public:
    PatchEmbed(int ps = 4, int ic = 3, int ed= 96, bool has_norm = false) : 
            patch_size(ps), in_chan(ic), embed_dim(ed), has_norm(has_norm) { 
        proj = register_module("proj", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_chan, embed_dim, patch_size).stride(patch_size)));
        if (has_norm)
            norm = register_module("norm", torch::nn::LayerNorm(ed));
    }
    torch::Tensor forward(torch::Tensor);

private:
    int patch_size;
    int in_chan;
    int embed_dim;
    bool has_norm;
    torch::nn::Conv2d proj;
    torch::nn::LayerNorm norm = nullptr;
};

torch::Tensor PatchEmbed::forward(torch::Tensor x) {
    int H = int(x.sizes()[2]);
    int W = int(x.sizes()[3]);
    if (W % patch_size != 0)
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, patch_size - W % patch_size}));
    if (H % patch_size != 0)
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, patch_size - H % patch_size}));
    
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
public:
    SwinTransformer(int, int, int, int, std::vector<int>, std::vector<int>,
                    int, float, bool, float, float, float, float, //torch::nn::LayerNorm,
                    bool, bool, std::vector<int>, int, bool);
    
    void freeze_stages();
    void init_weights(std::string &pretrained);
    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor x);
    void Train(bool mode = true) {
        this->train(mode);
        freeze_stages();
    }

    int pretrain_img_size, num_layers, embed_dim, frozen_stages;
    bool ape, patch_norm ;
    float drop_rate;
    std::vector<int> out_ind, num_heads, num_features;
    PatchEmbed patch_embed;
    torch::nn::Dropout pos_drop = nullptr;
};

SwinTransformer::SwinTransformer(int pis = 224, int patch_size = 4, int in_chans = 3, int ed = 96,
                                 std::vector<int> depths = {2, 2, 6, 2}, std::vector<int> num_heads = {3, 6, 12, 24},
                                 int window_size = 7, float mlp_ratio = 4.0, bool qkv_bias = true,
                                 float qk_scale = -1, float drop_rate = 0.0, float attn_drop_rate = 0.0, 
                                 float drop_path_rate = 0.2, //torch::nn::LayerNorm norm_layer = torch::nn::LayerNorm(), REMOVED PARAMETER BC THIS LAYER IS ALWAYS LAYERNORM
                                 bool ape = false, bool patch_norm = true, std::vector<int> out_indices = {0, 1, 2, 3},
                                 int frozen_stages = -1, bool use_checkpoint = false) : 
                                 pretrain_img_size(pis), num_layers(int(depths.size())), embed_dim(ed), ape(ape),
                                 patch_norm(patch_norm), out_ind(out_indices), frozen_stages(frozen_stages),
                                 drop_rate(drop_rate)
{
    patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, patch_norm);
    if (ape) {
        int patches_resolution = pretrain_img_size / patch_size;
        torch::Tensor &t = register_parameter("absolute_pos_embed", torch::zeros({1, embed_dim, patches_resolution, patches_resolution}));
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
    // std::vector<float> dpr = {x.item() for x in torch::linspace(0, drop_path_rate, depths_sum)};
    torch::Tensor dpr = torch::linspace(0, drop_path_rate, depths_sum);

    for (int i_layer; i_layer < num_layers; ++i_layer) {
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
        register_module("BasicLayer" + i_layer, layer);
    }
    
    for (int i = 0; i < num_layers; ++i)
        num_features.push_back(int(embed_dim * pow(2, i)));
    
    for (auto& i_layer : out_ind) {
        torch::nn::LayerNorm layer = torch::nn::LayerNorm(num_features[i_layer]);
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
            // layer.training();
            // for (auto &param : layer.parameters())
        }
    }
}

// void SwinTransformer::init_weights(std::string &pretrained) {

// }

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
    else
        x = x.flatten(2).transpose(1, 2);
    x = pos_drop(x);

    std::unordered_map<std::string, torch::Tensor> outs{};
    for (int i = 0; i < num_layers; ++i) {
        auto &layer = named_modules()["BasicLayer" + i];
        std::vector<torch::Tensor> res = static_cast<BasicLayer>(layer)(x, Wh, Ww);
        x = res[0];
        Wh = *res[1].select(0,0).data_ptr<int64_t>();
        Ww = *res[1].select(0,1).data_ptr<int64_t>();
        
        if (std::find(out_ind.begin(), out_ind.end(), i) != out_ind.end()) {
            auto &norm_layer = named_modules()["norm" + i];
            x = static_cast<torch::nn::LayerNorm>(norm_layer)(x);
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
                    float, float, float, float, //torch::nn::LayerNorm,
                    bool, bool, std::vector<int>, int, bool);
    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor);
    std::unordered_map<std::string, ShapeSpec> output_shape();
    inline int size_divisibility() const { return 32; }

private:
    std::vector<std::string> out_features;
    SwinTransformer swintransformer;
    std::unordered_map<std::string, int> out_feature_strides{};
    std::unordered_map<std::string, int> out_feature_channels{};
};

D2SwinTransformer::D2SwinTransformer(std::vector<std::string> out_features, int pis = 224, int patch_size = 4, int in_chans = 3, int ed = 96,
                                    std::vector<int> depths = {2, 2, 6, 2}, std::vector<int> num_heads = {3, 6, 12, 24},
                                    int window_size = 7, float mlp_ratio = 4.0, bool qkv_bias = true,
                                    float qk_scale = -1, float drop_rate = 0.0, float attn_drop_rate = 0.0, 
                                    float drop_path_rate = 0.2, //torch::nn::LayerNorm norm_layer = torch::nn::LayerNorm(),
                                    bool ape = false, bool patch_norm = true, std::vector<int> out_indices = {0, 1, 2, 3},
                                    int frozen_stages = -1, bool use_checkpoint = false) : out_features(out_features)
{
    swintransformer = SwinTransformer(pis, patch_size, in_chans, ed, depths, num_heads, window_size,
                                      mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, //norm_layer, 
                                      ape, patch_norm, out_indices, frozen_stages, use_checkpoint);
    out_feature_strides.insert({{"res2", 4}, {"res3", 8}, {"res4", 16}, {"res5", 32}});
    out_feature_channels.insert({{"res2", swintransformer.num_features[0]},
                                 {"res3", swintransformer.num_features[1]},
                                 {"res4", swintransformer.num_features[2]},
                                 {"res5", swintransformer.num_features[3]}});
}

std::unordered_map<std::string, torch::Tensor> D2SwinTransformer::forward(torch::Tensor x) {
    if (x.dim() != 4)
        std::cerr << "SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!";
    
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


/* change:
    - classes to structs
    - refs in method parameters
*/ 