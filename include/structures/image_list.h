#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef IMAGELIST_H
#define IMAGELIST_H


class ImageList {
public:
    ImageList(torch::Tensor t) : tensor(t) { int image_sizes[3]{int(t.sizes()[0]), int(t.sizes()[1]), int(t.sizes()[2])};  }
    int len();
    torch::Tensor getitem(int idx) { return tensor[idx]; }
    void device(torch::Device d) { tensor.to(torch::TensorOptions().device(d)); }
    ImageList from_tensors(torch::Tensor tensors,
                           int size_divisibility=0,
                           float pad_value = 0.0
                           );

private:
    torch::Tensor tensor;
    int image_sizes[3];
};

#endif