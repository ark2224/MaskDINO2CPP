#include <iostream>
#include <torch/torch.h>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "image_list.h"
#include "memory.h"





class MaskDINO2Decoder : torch::nn::Module {
public:
    MaskDINO2Decoder from_config(CfgNode cfg);
    torch::Tensor dn_post_process(torch::Tensor outputs_class,
                                  torch::Tensor outputs_coord,
                                  std::unordered_map<std::string, torch::Tensor> mask_map,
                                  torch::Tensor outputs_mask);
    double get_valid_ratio(torch::Tensor mask);
    torch::Tensor pred_box(torch::Tensor reference, torch::Tensor hs, bool ref0 = false);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask_features,
                          torch::Tensor masks, torch::Tensor targets);
    std::vector<torch::Tensor> forward_prediction_heads(torch::Tensor output,
                                                        torch::Tensor mask_features,
                                                        bool pred_mask = true);
    void set_aux_loss(torch::Tensor outputs_class,
                      torch::Tensor outputs_seg_masks,
                      torch::Tensor out_boxes);

};