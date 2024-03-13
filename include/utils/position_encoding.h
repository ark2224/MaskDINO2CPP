#include <torch/torch.h>
#include <math.h>
#include <vector>
#ifndef POSITIONENCODING_H
#define POSITIONENCODING_H


class PositionEmbeddingSine : torch::nn::Module {
/*
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
*/
public:
    PositionEmbeddingSine(int const,
                          int const,
                          bool const,
                          double const);
    // class methods:
    torch::Tensor forward(torch::Tensor&,
                          torch::Tensor);
private:
    int num_pos_feats, temperature;
    bool normalize;
    double scale;
};

PositionEmbeddingSine::PositionEmbeddingSine(int const num_pos_feats = 64,
                                             int const temperature = 10000,
                                             bool const normalize = false,
                                             double const scale = 0.) : 
torch::nn::Module(), num_pos_feats(num_pos_feats), temperature(temperature), normalize(normalize)
{
    if (scale != 0 && !normalize)
        std::cerr << "normalize should be true if scale is passed";
    if (scale == 0)
        this->scale = 2 * M_PI;
    this->scale = scale;
}

torch::Tensor PositionEmbeddingSine::forward(torch::Tensor& x,
                                             torch::Tensor mask = torch::Tensor())
{
    if (!mask.defined())
        mask = torch::zeros({x.sizes()[0], x.sizes()[2], x.sizes()[3]},
                            torch::TensorOptions().device(x.device()).dtype(torch::kBool));
    torch::Tensor not_mask = ~mask;
    torch::Tensor y_embed = not_mask.cumsum(1, torch::kFloat32);
    torch::Tensor x_embed = not_mask.cumsum(2, torch::kFloat32);
    if (this->normalize) {
        double eps = 1e-6;
        y_embed = y_embed / (y_embed.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(-1, torch::indexing::None),
                                           torch::indexing::Slice()})
                                           + eps) * this->scale;
        x_embed = x_embed / (x_embed.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(-1, torch::indexing::None)})
                                           + eps) * this->scale;
    }
    torch::Tensor dimension_t = torch::arange(this->num_pos_feats,
                                              torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    dimension_t = pow(this->temperature,
                      (2 * torch::floor_divide(dimension_t, 2) / this->num_pos_feats));

    torch::Tensor pos_x = x_embed.index({torch::indexing::Slice(),
                                        torch::indexing::Slice(),
                                        torch::indexing::Slice(),
                                        torch::indexing::Slice(torch::indexing::None)})
                                        / dimension_t;

    torch::Tensor pos_y = y_embed.index({torch::indexing::Slice(),
                                        torch::indexing::Slice(),
                                        torch::indexing::Slice(),
                                        torch::indexing::Slice(torch::indexing::None)})
                                        / dimension_t;

    pos_x = torch::stack({pos_x.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(0, torch::indexing::None, 2)}).sin(),
                          pos_x.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(1, torch::indexing::None, 2)}).cos()},
                          4).flatten(3);

    pos_y = torch::stack({pos_y.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(0, torch::indexing::None, 2)}).sin(),
                          pos_y.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(),
                                      torch::indexing::Slice(1, torch::indexing::None, 2)}).cos()},
                          4).flatten(3);

    torch::Tensor pos = torch::cat({pos_y, pos_x}, 3).permute({0, 3, 1, 2});
    return pos;
}


torch::Tensor gen_sineembed_for_position(torch::Tensor pos_tensor) {
/*
    n_query, bs, _ = pos_tensor.size()
    sineembed_tensor = torch.zeros(n_query, bs, 256)
*/
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

#endif
