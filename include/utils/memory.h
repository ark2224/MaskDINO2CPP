#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>

#ifndef MEMORY_H
#define MEMORY_H

void retry_if_cuda_oom(torch::Tensor *(torch::Tensor, torch::Tensor)) {
    void maybe_to_cpu(const torch::Tensor &x);
    void wrapped();
}

#endif