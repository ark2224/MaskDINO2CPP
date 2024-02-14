#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <array>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
int return_model(string s) {
    string model_file;
    if (s == "1") {
        model_file = "maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth";
    } else if (s == "2") {
        model_file = "maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth";
    }
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_file);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n" << endl;
        return -1;
    }
    return 0;
}

string return_option(const string options) {
    cout << options << endl;
    string opt;
    std::getline(cin, opt);
    while (opt != "1" && opt != "2") {
        cout << "Invalid input. Enter \'1\' or \'2\'.\n" << options << endl;
        std::getline(cin, opt);
    }
    return opt;
}


int main(int argc, const char* argv[]) {
    cout << "Deployment version of academic MaskDINO: Towards A Unified \
            Transformer-based Framework for Object Detection and \
            Segmentation[1]. The model is meant for panoptic \
            segmentation and is currently one of the best performers \
            for tackling this vision task. My deployment program \
            provides two options: to load and run a pretrained model \
            or to train a MaskDINO from scratch. \n" << endl;
    "Select your MaskDINO deployment option:\n\
    [1] Use Pretrained Model\n\
    [2] Train New Model\n";
    
    string options = "Select your MaskDINO deployment option:\n\
    [1] Use Pretrained Model\n\
    [2] Train New Model\n";

    string opt = return_option(options);
    if (opt == "1") {
        options = "Select backbone for pretrained model:\n\
        [1] SwinL Transformer \n[2] Resnet50";
        opt = return_option(options);
        return_model(opt);
        return 0;
    }
    else if (opt == "2") {
        return 0;
    }

    return 0;
}

