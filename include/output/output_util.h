#include <torch/torch.h>
#include <math.h>
#include <vector>

#ifndef OUTPUT_UTIL_H
#define OUTPUT_UTIL_H



std::vector<torch::Tensor> gen_encoder_output_proposals(
    torch::Tensor& memory,
    torch::Tensor& memory_padding_mask,
    std::vector< std::pair<int64_t, int64_t> >& spatial_shapes
) {
/*
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4

*/
    int N_ = (int)memory.sizes()[0];
    int S_ = (int)memory.sizes()[1];
    int C_ = (int)memory.sizes()[2];
    float base_scale = 4.0;
    std::vector<torch::Tensor> proposals{};
    int _cur = 0, idx = 0;
    for (auto& shape : spatial_shapes) {
        int64_t H_ = shape.first; 
        int64_t W_ = shape.second;
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


torch::Tensor inverse_sigmoid(torch::Tensor x, double eps = 1e-5) {
    x = x.clamp(0, 1);
    torch::Tensor x1 = x.clamp(eps);
    torch::Tensor x2 = (1 - x).clamp(eps);
    return torch::log(x1 / x2);
}

#endif