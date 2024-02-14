#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef IMAGELIST_H
#define IMAGELIST_H


class ImageList {
public:
    ImageList(torch::Tensor t) : tensor(t) { int image_sizes[3]{t.sizes()[0], t.sizes()[1], t.sizes()[2]};  }
    // ImageList(torch::Tensor t, int arr[3]) : tensor(t), image_sizes(arr) { }
    int len();
    torch::Tensor getitem(int idx) { return tensor[idx]; }
    ImageList to();
    bool device();
    ImageList from_tensors(torch::Tensor tensors,
                           int size_divisibility=0,
                           float pad_value = 0.0
                        //    std::unordered_map<std::string,int> padding_constraints
                           );

private:
    torch::Tensor tensor;
    int image_sizes[3];
};

#endif