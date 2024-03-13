#pragma once
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "cfg.h"
#ifndef DETECTRON2_H
#define DETECTRON2_H

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

#endif
