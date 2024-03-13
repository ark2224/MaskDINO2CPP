#include <iostream>
#include "cfg.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#ifndef BACKBONE_H
#define BACKBONE_H


struct Backbone : torch::nn::Module {
/* Abstract base class for network backbones. */
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

    std::string NAME;
};

#endif