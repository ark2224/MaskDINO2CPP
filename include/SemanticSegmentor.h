#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "cfg.h"
#include "registry.h"
#include "backbone.h"
#ifndef SEMANTICSEGMENTOR_H
#define SEMANTICSEGMENTOR_H

class SemanticSegmentor : torch::nn::Module {
public:
    SemanticSegmentor(Backbone b, torch::nn::Module ssh,
                      std::vector<float> pm, std::vector<float> ps) : 
                      backbone(b), sem_seg_head(ssh), pixel_mean(pm),
                      pixel_std(ps) { }
    SemanticSegmentor from_config(const CfgNode, const ShapeSpec);
    void device();
    std::unordered_map<std::string, torch::Tensor> forward(torch::Tensor&) const;
    // torch::Tensor forward(torch::Tensor &batch_inputs) const;

private:
    Backbone backbone;
    torch::nn::Module sem_seg_head;
    std::vector<float> pixel_mean, pixel_std;
};

SemanticSegmentor SemanticSegmentor::from_config(const CfgNode cfg, const ShapeSpec input_shape) {
    backbone = build_backbone(cfg);

    return *this;
}

// only has a temporary body:
std::unordered_map<std::string, torch::Tensor>
    SemanticSegmentor::forward(torch::Tensor &batch_inputs) const {
        std::unordered_map<std::string, torch::Tensor> tmp = {{"pred_logits", batch_inputs},
                                                              {"pred_masks", batch_inputs},
                                                              {"pred_boxes", batch_inputs}};
        // std::vector< std::unordered_map<std::string, torch::Tensor> > vec{tmp};
        return tmp;
}
// torch::Tensor SemanticSegmentor::forward(torch::Tensor &batch_inputs) const {
//     return batch_inputs;
// }

// NEED TO FINISH FOR THIS METHOD: make registry class @detectron2.utils.registry import Registry
SemanticSegmentor build_semantic_seg_head(const CfgNode cfg, const ShapeSpec *input_shape = nullptr) {
    SEM_SEG_HEADS_REGISTRY.doc("Registry for semantic segmentation heads, which make semantic segmentation predictions from feature maps.");
    // std::string name = cfg.MODEL.SEM_SEG_HEAD.NAME; //NEEDS TO BE FIXED. CAN'T BE ATTRIBUTE MODEL->SEM_SEG_HEAD, MODEL IS ONLY A WRAPPER

    std::string name = cfg.getattr("MODEL").front();

    return (*(dynamic_cast<SemanticSegmentor*>(SEM_SEG_HEADS_REGISTRY.get(name)))).from_config(cfg, *input_shape);
}

torch::Tensor sem_seg_postprocessing(torch::Tensor &result, std::vector<int64_t> &img_size, int &output_height, int &output_width) {
    return result;
}

#endif