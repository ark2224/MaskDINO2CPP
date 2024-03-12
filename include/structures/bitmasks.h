#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "boxes.h"
#include "instances.h"

#ifndef BITMASKS_H
#define BITMASKS_H


class BitMasks : torch::Tensor {
public:
    BitMasks() : tensor(torch::Tensor()) { }
    BitMasks(torch::Tensor& x) : tensor(x) { }
    BitMasks to();
    bool device();
    BitMasks getitem(const torch::Tensor&);
    torch::Tensor *iter() { return &tensor; }
    std::string str() const;
    size_t len() const { return (*this).sizes()[0]; };
    torch::Tensor nonempty();
    // BitMasks from_polygon_masks();
    // BitMasks from_roi_masks();
    torch::Tensor crop_and_resize(torch::Tensor &boxes, int (&mask_size)[2]);
    BitMasks get_bounding_boxes();
    static BitMasks cat(std::vector<BitMasks> &);

    torch::Tensor tensor;
};


#endif