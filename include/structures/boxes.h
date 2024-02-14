#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef BOXES_H
#define BOXES_H

class Boxes : torch::Tensor {
public:
    Boxes(torch::Tensor t) : tensor(t) { }
    Boxes(Boxes &b) : tensor(b.tensor) { }
    Boxes* clone() const { return new Boxes(*this); }
    bool to(std::string device);
    torch::Tensor area();
    void clip(int box_size[2]);
    torch::Tensor nonempty(float threshold = 0.0);
    Boxes getitem(torch::Tensor item);
    // int len() { return tensor.sizes()[0]; }
    std::string repr();
    torch::Tensor inside_box(int box_size[2], int boundary_threshold = 0);
    torch::Tensor get_centers();
    void scale(float scale_x, float scale_y);
    Boxes cat(std::vector<Boxes> boxes_vec);
    bool device(std::string);
    void *iter() { return tensor.data_ptr(); }

private:
    torch::Tensor tensor;
};

#endif