#include <torch/torch.h>
#include <math.h>
#include <vector>
#ifndef MODULES_H
#define MODULES_H

torch::nn::Sequential get_clones(
    torch::nn::AnyModule& module,
    int const& num_layers,
    bool const& layer_share = false
) {
    torch::nn::Sequential ret = torch::nn::Sequential();
    if (layer_share) {
        for (int i = 0; i < num_layers; ++i)
            ret->push_back(module);
        return ret;
    }
    else {
        for (int i = 0; i < num_layers; ++i)
            ret->push_back(module.clone());
        return ret;
    }
}


class MLPImpl : torch::nn::Module {
    /* Multi-layer Perceptron */
public:
    MLPImpl() { } // default constructor for SwinTransformerBlock declaration
    MLPImpl(int in_features, int hidden_features, int out_features, float d = 0.0) : 
        torch::nn::Module(), fc1(in_features, hidden_features), fc2(hidden_features, out_features), drop(d) { }
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


class DropPath : torch::nn::Module {
/*
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
*/
public:
    DropPath() { }
    DropPath(float dp, bool sbk = true, bool t = false) :
             drop_prob(dp), scale_by_keep(sbk), training(t) { }
    torch::Tensor drop_path(torch::Tensor, float, bool, bool);
    torch::Tensor forward(torch::Tensor x) { 
        return drop_path(x, drop_prob, training, scale_by_keep);
    }

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

#endif
