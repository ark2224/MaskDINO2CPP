#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef BOXOPERATIONS_H
#define BOXOPERATIONS_H

/*
    Utilities for bounding boxes and their manipulation
*/

torch::Tensor box_cxcywh_to_xyxy(torch::Tensor &x) {
    std::vector<torch::Tensor> vec = x.unbind(-1);
    if (vec.size() != 4)
        std::cerr << "input tensor to box_xyxy_to_cxcywh does not have 4 dimensions";
    torch::Tensor x_c = vec[0];
    torch::Tensor y_c = vec[1];
    torch::Tensor w = vec[2];
    torch::Tensor h = vec[3];

    std::vector<torch::Tensor> b = {
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * w
    };

    return torch::stack(b, -1);
}


torch::Tensor box_xyxy_to_cxcywh(torch::Tensor &x) {
    std::vector<torch::Tensor> vec = x.unbind(-1);
    if (vec.size() != 4)
        std::cerr << "input tensor to box_xyxy_to_cxcywh does not have 4 dimensions";
    torch::Tensor x0 = vec[0];
    torch::Tensor y0 = vec[1];
    torch::Tensor x1 = vec[2];
    torch::Tensor y1 = vec[3];

    std::vector<torch::Tensor> b = {
        (x0 + x1) * 0.5,
        (y0 + y1) * 0.5,
        (x1 - x0),
        (y1 - y0),
    };

    return torch::stack(b, -1);
}


torch::Tensor masks_to_boxes(torch::Tensor &masks) {
/*
    Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
*/
    if (masks.numel() == 0)
        return torch::zeros({0, 4}, torch::TensorOptions().device(masks.device()));
    
    int h = (int)masks.sizes()[-2];
    int w = (int)masks.sizes()[-1];

    torch::Tensor y = torch::arange(0, h, torch::TensorOptions().dtype(torch::kFloat).device(masks.device()));
    torch::Tensor x = torch::arange(0, w, torch::TensorOptions().dtype(torch::kFloat).device(masks.device()));
    std::vector<torch::Tensor> grid = torch::meshgrid({y, x});

    torch::Tensor x_mask = (masks * x.unsqueeze(0));
    torch::Tensor x_max = std::get<0>(x_mask.flatten(1).max(-1));
    torch::Tensor x_min = std::get<0>(x_mask.masked_fill(~(masks.to(torch::kBool)), 1e8).flatten(1).min(-1));

    torch::Tensor y_mask = (masks * y.unsqueeze(0));
    torch::Tensor y_max = std::get<0>(y_mask.flatten(1).max(-1));
    torch::Tensor y_min = std::get<0>(y_mask.masked_fill(~(masks.to(torch::kBool)), 1e8).flatten(1).min(-1));

    return torch::stack({x_min, y_min, x_max, y_max}, 1);
}



torch::Tensor window_partition(torch::Tensor x, int window_size) {
/*
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
*/
    int B = int(x.sizes()[0]);
    int H = int(x.sizes()[1]);
    int W = int(x.sizes()[2]);
    int C = int(x.sizes()[3]);
    x = x.view({B, int(H / window_size), window_size, int(W / window_size), window_size, C});
    torch::Tensor windows = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({-1, window_size, window_size, C});
    return windows;
}

torch::Tensor window_reverse(torch::Tensor windows, int window_size, int H, int W) {
/*
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
*/
    int B = int(windows.sizes()[0] / (H * W / window_size / window_size));
    torch::Tensor x = windows.view({B, int(H / window_size), int(W / window_size), window_size, window_size, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous().view({B, H, W, -1});
    return x;
}


class WindowAttention : torch::nn::Module {
/*
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0

*/
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
/*
    Forward function.
    Args:
        x: input features with shape of (num_windows*B, N, C)
        mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

*/
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

#endif