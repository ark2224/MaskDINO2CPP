#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "boxes.h"
#include "instances.h"

#ifndef BITMASKS_H
#define BITMASKS_H


class BitMasks : std::vector<Instances> {
public:
    BitMasks to();
    bool device();
    BitMasks getitem(const torch::Tensor&);
    torch::Tensor *iter() { return &tensor; }
    std::string str() const;
    size_t len() const { return (*this).size(); };
    torch::Tensor nonempty();
    // BitMasks from_polygon_masks();
    // BitMasks from_roi_masks();
    torch::Tensor crop_and_resize(torch::Tensor &boxes, int (&mask_size)[2]);
    Boxes get_bounding_boxes();
    static BitMasks cat(std::vector<BitMasks> &);
private:
    torch::Tensor tensor;
};


#endif