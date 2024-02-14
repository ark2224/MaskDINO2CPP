#include <iostream>
#include "cfg.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include "registry.h"
#ifndef BACKBONE_H
#define BACKBONE_H


static Registry BACKBONE_REGISTRY = Registry("BACKBONE");
static Registry SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS");


struct Backbone : torch::nn::Module {
/* Abstract base class for network backbones. */
    Backbone from_config(const CfgNode, const ShapeSpec) const;

    // std::vector<torch::Tensor> forward(torch::Tensor) const;
    torch::Tensor forward(torch::Tensor &) const;
    /* Subclasses must override this method, but adhere to the same return type.
    Returns:
        vector[str->Tensor]: mapping from feature name (e.g., "res2") to tensor */

    int size_divisibility() const { return 0; }
    /*  Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.*/

    std::vector<std::string> padding_constraints() 
        { std::vector<std::string> v; return v; }
    /*  This property is a generalization of size_divisibility. Some backbones and training
        recipes require specific padding constraints, such as enforcing divisibility by a specific
        integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
        in :paper:vitdet). `padding_constraints` contains these optional items like:
        {
            "size_divisibility": int,
            "square_size": int,
            # Future options are possible
        }
        `size_divisibility` will read from here if presented and `square_size` indicates the
        square padding size if `square_size` > 0.*/

    ShapeSpec output_shape() const { ShapeSpec v; return v; }

    std::string NAME;
};

// FIGURE OUT WHAT TO DO FOR THIS!!!!!!!!!!!!!!!!!!
Backbone Backbone::from_config(const CfgNode, const ShapeSpec) const {
    return *this;
}

// std::vector<torch::Tensor> Backbone::forward(torch::Tensor x) const {
torch::Tensor Backbone::forward(torch::Tensor &x) const {
    // return std::vector<torch::Tensor>{x};
    return x;
}



// FINISH BACKBONE_REGISTRY PART
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/backbone.py
Backbone build_backbone(const CfgNode cfg, ShapeSpec *input_shape = nullptr) {
/*  Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone` */
    std::vector<std::string> PIXEL_MEAN = cfg.getattr("PIXEL_MEAN");
    ShapeSpec tmp = ShapeSpec(static_cast<int>(PIXEL_MEAN.size()));
    if (input_shape == nullptr) {
        input_shape = &tmp;
    }
    // std::string backbone_name = cfg.MODEL.BACKBONE.NAME;
    std::string backbone_name = cfg.getattr("BACKBONE").front();

    // fix the stuff below:
    Backbone backbone = (*(dynamic_cast<Backbone*>(BACKBONE_REGISTRY.get(backbone_name)))).from_config(cfg, *input_shape);
    if (typeid(backbone_name).name() != "Backbone") 
        std::cout << "Backbone not built correctly. Please fix." << std::endl; 
    return backbone;
};

#endif