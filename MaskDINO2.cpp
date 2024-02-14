#include <iostream>
#include <torch/torch.h>
#include "backbone.h"
#include "matcher.h"
#include "SemanticSegmentor.h"
#include "criterion.h"
#include "image_list.h"
#include "memory.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;
class MaskDINO2 : torch::nn::Module {
public:
// Two Basic Constructors:
    MaskDINO2(Backbone bb,
              SemanticSegmentor ssh, 
              SetCriterion crit,
              int nq,
              float ot,
              float omt,
              vector<string> md,
              std::vector<int> sd,
              bool sspbi,
              float pm,
              float ps,
              // Additional args for inference:
              bool so,
              bool po,
              bool io,
              int ttpi,
              string dl,
              float pt,
              bool fob,
              bool te,
              bool scl) : 
              backbone(bb),
              semantic_seg_head(ssh), 
              criterion(crit),
              num_queries(nq),
              overlap_threshold(ot),
              object_mask_threshold(omt),
              metadata(md),
              size_divisibility(sd),
              sem_seg_postprocess_before_inference(sspbi),
              pixel_mean(pm),
              pixel_std(ps),
              // Additional args for inference:
              semantic_on(so),
              panoptic_on(po),
              instance_on(io),
              test_topk_per_image(ttpi),
              data_loader(dl),
              pano_temp(pt),
              focus_on_box(fob),
              transform_eval(te),
              semantic_ce_loss(scl) { }
    
    MaskDINO2(Backbone bb,
              SemanticSegmentor ssh, 
              SetCriterion crit,
              int nq,
              float ot,
              float omt,
              vector<string> md,
              std::vector<int> sd,
              bool sspbi,
              float pm,
              float ps) : 
              backbone(bb),
              semantic_seg_head(ssh), 
              criterion(crit),
              num_queries(nq),
              overlap_threshold(ot),
              object_mask_threshold(omt),
              metadata(md),
              size_divisibility(sd),
              sem_seg_postprocess_before_inference(sspbi),
              pixel_mean(pm),
              pixel_std(ps) { }

    MaskDINO2 *from_config(CfgNode); //should you have this?
    
    std::vector<std::unordered_map< std::string, torch::Tensor >> forward(vector<vector<torch::Tensor>> batched_inputs);
    /*Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
    */

    vector<torch::Tensor> prepare_targets(torch::Tensor &targets, torch::Tensor &images);

    vector<torch::Tensor> prepare_targets_detr(torch::Tensor targets, torch::Tensor images);

    torch::Tensor sematnic_inference(torch::Tensor mask_cls, torch::Tensor mask_pred);

    torch::Tensor panoptic_inference(torch::Tensor &mask_cls, torch::Tensor &mask_pred);//check return type

    torch::Tensor instance_inference(torch::Tensor mask_cls, torch::Tensor mask_pred, torch::Tensor mask_box_result);//check return type

    torch::Tensor box_postprocess(torch::Tensor out_bbox, int img_h, int img_w);

private:
/*  Args:
        backbone: a backbone module, must follow detectron2's backbone interface
        sem_seg_head: a module that predicts semantic segmentation from backbone features
        criterion: a module that defines the loss
        num_queries: int, number of queries
        object_mask_threshold: float, threshold to filter query based on classification score
            for panoptic segmentation inference
        overlap_threshold: overlap threshold used in general inference for panoptic segmentation
        metadata: dataset meta, get `thing` and `stuff` category names for panoptic
            segmentation inference
        size_divisibility: Some backbones require the input height and width to be divisible by a
            specific integer. We can use this to override such requirement.
        sem_seg_postprocess_before_inference: whether to resize the prediction back
            to original input size before semantic segmentation inference or after.
            For high-resolution dataset like Mapillary, resizing predictions before
            inference will cause OOM error.
        pixel_mean, pixel_std: list or tuple with #channels element, representing
            the per-channel mean and std to be used to normalize the input image
        semantic_on: bool, whether to output semantic segmentation prediction
        instance_on: bool, whether to output instance segmentation prediction
        panoptic_on: bool, whether to output panoptic segmentation prediction
        test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        transform_eval: transform sigmoid score into softmax score to make score sharper
        semantic_ce_loss: whether use cross-entroy loss in classification
*/
    Backbone backbone;
    SemanticSegmentor semantic_seg_head;
    SetCriterion criterion;
    int num_queries;
    float overlap_threshold;
    float object_mask_threshold;
    vector<string> metadata;
    std::vector<int> size_divisibility;
    bool sem_seg_postprocess_before_inference;
    float pixel_mean;//meant to be a Tuple; change to vector of whatever size
    float pixel_std;//meant to be a Tuple; change to vector of whatever size
    bool training = false;
    
    // Additional args for inference:
    bool semantic_on = false;
    bool panoptic_on = false;
    bool instance_on = false;
    int test_topk_per_image;
    string data_loader;
    float pano_temp;
    bool focus_on_box = false;
    bool transform_eval = false;
    bool semantic_ce_loss = false;
};

MaskDINO2 *MaskDINO2::from_config(CfgNode cfg) { //should you have this?
    backbone = build_backbone(cfg);
    semantic_seg_head = build_semantic_seg_head(cfg, &backbone.output_shape());

    const std::unordered_map< std::string, float> &model_vec = cfg.contents.at("MODEL");

    // loss parameters:
    bool deep_supervision = model_vec.at("deep_supervision");
    bool no_object_weight = model_vec.at("no_object_weight");

    // loss weights:
    float class_weight = model_vec.at("class_weight");
    float cost_class_weight = model_vec.at("cost_class_weight ");
    float cost_dice_weight = model_vec.at("cost_dice_weight ");
    float dice_weight = model_vec.at("dice_weight ");
    float cost_mask_weight = model_vec.at("cost_mask_weight");
    float mask_weight = model_vec.at("mask_weight");
    float cost_box_weight = model_vec.at("cost_box_weight");
    float box_weight = model_vec.at("box_weight");
    float cost_giou_weight = model_vec.at("cost_giou_weight");
    float giou_weight = model_vec.at("giou_weight");

    HungarianMatcher matcher = HungarianMatcher(
        cost_class_weight,
        cost_mask_weight,
        cost_dice_weight,
        model_vec.at("train_num_points"),
        cost_box_weight,
        cost_giou_weight
    );

    criterion.weight_dict = {{"loss_ce", class_weight},
                             {"loss_mask", mask_weight},
                             {"loss_dice", dice_weight},
                             {"loss_box", box_weight},
                             {"loss_giou", giou_weight}};

    if (model_vec.at("two_stage")) {
        std::unordered_map<std::string, float> interim_weight_dict{};
        for (auto &[k, v] : criterion.weight_dict) {
            interim_weight_dict[k + "_interim"] = v;
        }
        criterion.weight_dict = interim_weight_dict;
    }

    // denoising training
    float dn = model_vec.at("dn");
    if (dn == 0) { //"standard"
        std::unordered_map<std::string, float> interim_weight_dict{};
        for (auto &[k, v] : criterion.weight_dict) {
            interim_weight_dict[k + "_dn"] = v;
        }
        criterion.weight_dict = interim_weight_dict;
        criterion.dn_losses = {"labels", "boxes"};
    } else if (dn == 1) { //"seg"
        std::unordered_map<std::string, float> interim_weight_dict{};
        for (auto &[k, v] : criterion.weight_dict) {
            interim_weight_dict[k + "_dn"] = v;
        }
        criterion.weight_dict = interim_weight_dict;
        criterion.dn_losses = {"labels", "masks", "boxes"};
    } else {
        criterion.dn_losses = {};
    }
    
    if (deep_supervision) {
        float dec_layers = model_vec.at("dec_layers");
        std::unordered_map<std::string, float> aux_weight_dict = {};
        // =================================================LEFT OFF HERE FOR CONFIG=================================================
    }

    return this;
}

std::vector<std::unordered_map< std::string, torch::Tensor >>
                MaskDINO2::forward(vector<vector<torch::Tensor>> batched_inputs) {
    // ARGS:
    // batched_inputs : vector<vector<torch::Tensor>>
    //      each vector<vector<>> is a batch
    //      each vector<torch::Tensor> is an image (index 0) and an instance:
    //          "image": Tensor, image in (C, H, W) format.
    //          "instances": per-region ground truth
    std::vector<torch::Tensor> images;
    // ImageList img_lst = ImageList();
    for (vector<torch::Tensor> &ii : batched_inputs) {
        torch::Tensor x = (ii[0] - pixel_mean) / pixel_std;
        images.push_back(x);
    }
    // skipping the part that they would use an ImageList for images

    std::vector<torch::Tensor> features;
    for (auto &t : images) {
        features.push_back(backbone.forward(t));
    }

    if (training) {
        ;
        // will fill in later
    }
    else {
        std::vector< std::unordered_map<std::string, torch::Tensor> > outputs;
        std::vector<torch::Tensor> mask_cls_results, mask_pred_results, mask_box_results;

        std::vector<int64_t> sizes = {images[0].size(-2), images[0].size(-1)};
        torch::nn::functional::InterpolateFuncOptions options =
            torch::nn::functional::InterpolateFuncOptions()
                    .size(sizes)
                    .mode(torch::kBilinear)
                    .align_corners(false);
        
        for (torch::Tensor &t : features) {
            std::unordered_map<std::string, torch::Tensor> out = semantic_seg_head.forward(t);
            outputs.push_back(out);
            mask_cls_results.push_back(out["pred_logits"]);
            mask_box_results.push_back(out["pred_boxes"]);
            // upsample
            mask_pred_results.push_back(
                torch::nn::functional::interpolate(out["pred_masks"], options)
            );
        }

        std::vector<std::unordered_map< std::string, torch::Tensor >> processed_results;

        for (int i = 0; i < images.size(); ++i) {
            int height = images[i].size(0);
            int width = images[i].size(1);
            // skipped real height^
            processed_results.push_back({});
            std::vector<int64_t> new_size{mask_pred_results[i].size(-2), mask_pred_results[i].size(-1)}; // padded size

            if (sem_seg_postprocess_before_inference) {
                mask_pred_results[i] = sem_seg_postprocessing(mask_pred_results[i], sizes, height, width);
            }

            // semantic segmentation inference
            if (semantic_on) {
                torch::Tensor r = sematnic_inference(mask_cls_results[i], mask_pred_results[i]);
                if (sem_seg_postprocess_before_inference) {
                    r = sem_seg_postprocessing(mask_pred_results[i], sizes, height, width);
                processed_results.push_back({{"sem_seg", r}});
            }

            // panoptic segmentation inference
            if (panoptic_on) {
                torch::Tensor panoptic_r = panoptic_inference(mask_cls_results[i], mask_pred_results[i]);
                processed_results.push_back({{"panoptic_seg", panoptic_r}});
            }

            // instance segmentation inference
            if (instance_on) {
                height = new_size[0]/sizes[0]*height;
                width = new_size[1]/sizes[1]*width;
                mask_box_results[i] = box_postprocess(mask_box_results[i], height, width);
                torch::Tensor instance_r = instance_inference(mask_cls_results[i],
                                                              mask_pred_results[i],
                                                              mask_box_results[i]);
                processed_results.push_back({{"instances", instance_r}});
            }
        
        return processed_results;
        }

    }
}


vector<torch::Tensor> MaskDINO2::prepare_targets(torch::Tensor &targets, torch::Tensor &images) {

}

vector<torch::Tensor> MaskDINO2::prepare_targets_detr(torch::Tensor targets, torch::Tensor images) {

}

torch::Tensor MaskDINO2::sematnic_inference(torch::Tensor mask_cls, torch::Tensor mask_pred) {

}

torch::Tensor MaskDINO2::panoptic_inference(torch::Tensor &mask_cls, torch::Tensor &mask_pred) {

}

torch::Tensor MaskDINO2::instance_inference(torch::Tensor mask_cls, torch::Tensor mask_pred, torch::Tensor mask_box_result) {//check return type

}

torch::Tensor MaskDINO2::box_postprocess(torch::Tensor out_bbox, int img_h, int img_w) {

}


int main()
{
    cout << "\n ========================= Working ========================= \n" << endl;
    detectron2().boo();

    return 0;
}