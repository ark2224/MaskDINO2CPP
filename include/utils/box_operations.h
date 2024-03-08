#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "../structures/boxes.h"
#include "../structures/bitmasks.h"

#ifndef BOXOPERATIONS_H
#define BOXOPERATIONS_H


torch::Tensor box_cxcywh_to_xyxy(torch::Tensor &x);


torch::Tensor box_xyxy_to_cxcywh(torch::Tensor &x);

// float (*box_iou)(Boxes &boxes1, Boxes &boxes2)[2];
auto box_iou(Boxes &boxes1, Boxes &boxes2) -> float (*)[2];


float generalized_box_iou(Boxes &boxes1, Boxes &boxes2);


auto box_iou_pairwise(Boxes &boxes1, Boxes &boxes2) -> float (*)[2];


float generalized_box_iou_pairwise(Boxes &boxes1, Boxes &boxes2);


torch::Tensor masks_to_boxes(BitMasks &masks);

#endif