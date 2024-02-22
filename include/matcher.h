#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "criterion.h"

#ifndef MATCHER_H
#define MATCHER_H


class HungarianMatcher {
public:
    HungarianMatcher(float cc, float cm, float cd, int np, float cb, float cg) :
                    cost_class(cc), cost_mask(cm), cost_dice(cd), num_points(np),
                    cost_box(cb), cost_giou(cg) { }
    torch::Tensor memory_efficient_forward(torch::Tensor outputs,
                                           torch::Tensor targets,
                                           std::string cost);
    std::vector<torch::Tensor> forward(torch::Tensor outputs,
                                       torch::Tensor targets,
                                       std::string cost);
    void repr(int repr_indent = 4);
    float batch_dice_loss(torch::Tensor outputs,
                          torch::Tensor targets);
    float batch_sigmoid_ce_loss(torch::Tensor outputs,
                                torch::Tensor targets);

private:
    float cost_class = 1.;
    float cost_mask = 1.;
    float cost_dice = 1.;
    int num_points = 0;
    float cost_box = 0.;
    float cost_giou = 0.;
    bool panoptic_on = false;
};

#endif