#pragma once
#include <iostream>
#include "cfg.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include "registry.h"


class detectron2 {
public:
    void boo() {
        std::cout << "Header file included!\n" << std::endl;
    }
};


struct ShapeSpec {
    /* A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules. */
    ShapeSpec(int c = 0, int h = 0, int w = 0, int s = 0) :
              channels(c), height(h), width(w), stride(s) { }
    int channels;
    int height;
    int width;
    int stride;
};



// DETECTR0N2:
// https://github.com/facebookresearch/detectron2/tree/main

// static Registry BACKBONE_REGISTRY = Registry("BACKBONE");
// static Registry SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS");





// // NEED TO FINISH: @detectron2.utils.registry import Registry
// // actually imported from here in fvcore: https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/registry.py 
// class Registry {
// public:
//     Registry(std::string s) : name({s}) { }
//     void set_doc(std::string s) { doc = s; };
// private:
//     std::vector<std::string> name;
//     std::string doc;
// };


// class SemanticSegmentor : torch::nn::Module {
// public:
//     SemanticSegmentor(Backbone b, torch::nn::Module ssh,
//                       std::vector<float> pm, std::vector<float> ps) : 
//                       backbone(b), sem_seg_head(ssh), pixel_mean(pm),
//                       pixel_std(ps) { }
//     SemanticSegmentor from_config(const CfgNode);
//     void device();
//     std::vector< std::unordered_map<int, torch::Tensor> > forward(torch::Tensor);
// private:
//     Backbone backbone;
//     torch::nn::Module sem_seg_head;
//     std::vector<float> pixel_mean, pixel_std;
// };

// SemanticSegmentor SemanticSegmentor::from_config(const CfgNode cfg) {
//     backbone = build_backbone(cfg);

//     return *this;
// }

// std::vector< std::unordered_map<int, torch::Tensor> >
//     SemanticSegmentor::forward(torch::Tensor batch_inputs) {

//     }

// // NEED TO FINISH FOR THIS METHOD: make registry class @detectron2.utils.registry import Registry
// SemanticSegmentor build_semantic_seg_head(const CfgNode cfg, const ShapeSpec input_shape) {
//     SEM_SEG_HEADS_REGISTRY.doc("Registry for semantic segmentation heads, which make semantic segmentation predictions from feature maps.");
//     std::string name = cfg.MODEL.SEM_SEG_HEAD.NAME; //NEEDS TO BE FIXED. CAN'T BE ATTRIBUTE MODEL->SEM_SEG_HEAD, MODEL IS ONLY A WRAPPER
//     return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape);
// }

/*TODO:
-  finish yaml dependencies (CfgNode and its members [merge_from_file(), load_yaml_with_base(), ...])

*/