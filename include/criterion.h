#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "structures/bitmasks.h"
#include "utils/box_operations.h"
#ifndef CRITERION_H
#define CRITERION_H

// computes loss for DETR
class SetCriterion : torch::nn::Module {
public:
    std::vector<float> loss_labels_ce(const torch::Tensor &outputs,
                                      const torch::Tensor &targets,
                                      const int (&indices)[2],
                                      const int &num_masks);
    std::vector<float> loss_labels(const torch::Tensor &outputs,
                                   const torch::Tensor &targets,
                                   const int (&indices)[2],
                                   const int &num_masks,
                                   bool log=true);
    std::vector<float> loss_boxes(const torch::Tensor &outputs,
                                  const torch::Tensor &targets,
                                  const int (&indices)[2],
                                  const int &num_masks);
    std::vector<float> loss_masks(const torch::Tensor &outputs,
                                      const torch::Tensor &targets,
                                      const int (&indices)[2],
                                      const int &num_masks);
    void prep_for_dn(const std::vector<BitMasks> &mask_vec);
    void get_src_permutation_idx(const int (&indices)[2]);
    void get_target_permutation_idx(const int (&indices)[2]);
    std::vector<float> get_loss(const torch::Tensor &outputs,
                                const torch::Tensor &targets,
                                const int (&indices)[2],
                                const int &num_masks);
    void forward(const torch::Tensor &outputs,
                 const torch::Tensor &targets,
                 const int (&indices)[2],
                 const std::vector<BitMasks> *mask_vec = nullptr);
    void repr();

    int num_classes;
    torch::nn::Module matcher;
    std::unordered_map<std::string, float> weight_dict;
    float eos_coef;
    std::vector<float> losses;
    std::string dn;
    std::vector<std::string> dn_losses;
    // register_buffer(empty_weight);
    int num_points;
    float oversample_ratio;
    float importance_sample_ratio;
    float focal_alpha = 0.25;
    bool panoptic_on;
    std::vector<float> semantic_ce_loss;
};


class NestedTensor {
public:
    NestedTensor to(std::string device);
    void decompose();
    void repr();
private:
    torch::Tensor tensor, mask;
};


int get_world_size();


torch::Tensor point_sample(torch::Tensor &input, const torch::Tensor point_coords);


torch::Tensor get_uncertain_point_coords_with_randomness(torch::Tensor coarse_logits,
                                                         float uncertainty_func, //gonna need to change this to func* pointer
                                                         const int &num_points,
                                                         float oversample_ratio,
                                                         float importance_sample_ratio);


NestedTensor nested_tensor_from_tensor_vector(std::vector<torch::Tensor> tensor_vec);


bool is_dist_avail_and_initialized();

#endif