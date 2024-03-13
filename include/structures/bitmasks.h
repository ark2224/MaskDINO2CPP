#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef BITMASKS_H
#define BITMASKS_H


class BitMasks : torch::Tensor {
public:
    BitMasks() : tensor(torch::Tensor()) { }
    BitMasks(torch::Tensor& x) : tensor(x), img_size(x.sizes()) { }
    BitMasks get_bounding_boxes();

// class variables
    torch::Tensor tensor;
    c10::IntArrayRef img_size;
};

BitMasks BitMasks::get_bounding_boxes() {
    torch::Tensor boxes = torch::zeros({tensor.sizes()[0], 4}, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor x_any = torch::any(tensor, 1);
    torch::Tensor y_any = torch::any(tensor, 2);
    for (int idx = 0; idx < (int)tensor.sizes()[0]; ++idx) {
        torch::Tensor x = torch::where(x_any.index({idx, torch::indexing::Slice()}))[0];
        torch::Tensor y = torch::where(y_any.index({idx, torch::indexing::Slice()}))[0];

        if (x.sizes()[0] && y.sizes()[0]) {
            std::vector<float> vec = {*x.index({0}).data_ptr<float>(),
                                      *y.index({0}).data_ptr<float>(),
                                      *x.index({-1}).data_ptr<float>() + 1,
                                      *y.index({-1}).data_ptr<float>() + 1};
            boxes.index_put_(
                {idx, torch::indexing::Slice()},
                torch::from_blob(vec.data(),
                                {4},
                                torch::TensorOptions().dtype(torch::kFloat32))
            );
        }
    }
    return BitMasks(boxes);
}


#endif