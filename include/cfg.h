#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "detectron2.h"
#include "utils/pathmanager.h"
#ifndef CFG_H
#define CFG_H


void configurable() {
    bool wrapped();

    // void wrapper() {
    //     bool wrapped();
    // }

}

// // https://github.com/facebookresearch/iopath/blob/main/iopath/common/file_io.py
// // https://github.com/facebookresearch/fairseq/blob/main/fairseq/file_io.py
// class PathManager {
// public:
//     bool isfile(std::string);
//     void open(std::string, std::string);
// };

// bool PathManager::isfile(std::string filename) {
//     return false;
// }

// void PathManager::open(std::string filename, std::string opt = "r") {
//     ;
// }



// CFGNODE CODE BASED ON:
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/config.py#L87 
// https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/config.py
// both above are inheriting attributes from this original class: https://github.com/rbgirshick/yacs/blob/master/yacs/config.py
/*
ideas for implementation
    - use the keyword 'virtual' for functions. this class is simply an Interface: https://stackoverflow.com/questions/12902751/how-to-clone-object-in-c-or-is-there-another-solution
*/
// class CfgNode : std::unordered_map< std::string, std::vector<std::string> > {
class CfgNode {
public:
    CfgNode(const CfgNode* cfgn, const std::vector<std::string> kl) :
            key_list(kl) { for ( auto ptr : (*cfgn).children) {
                // unsure what to do for constructor!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            } }

    void open_cfg(std::string filename) {
        pathman.open(filename, "r");
    }
    void merge_from_file(std::string, bool);
    CfgNode load_yaml_with_base(std::string, bool);
    CfgNode merge_a_into_b(CfgNode a, CfgNode b);
    CfgNode load_with_base(std::string cfg_base_filename);
    void dump();//idk if i need this one...
    CfgNode merge_from_other_cfg(CfgNode cfg_other);

    // IDK WHAT TO PUT INSIDE VECTOR PARAMETER BELOW!!!
    CfgNode merge_from_vector(std::vector<std::string> cfg_vector); // adaptation from original method: merge_from_list(List cfg_list)

    void setattr(const std::string name,
                 const std::string val);
    CfgNode create_config_tree_from_dict(const std::vector<std::string> values,
                                         const std::vector<std::string> key_list);
    std::vector<std::string> getattr(const std::string name) const;
    void str() const;
    void repr() const;
    void freeze();
    void defrost();
    bool is_frozen() const;
    void immutable(bool is_immutable);
    CfgNode clone() const;
    void register_deprecated_key(const std::string old_name,
                                 const std::string new_name,
                                 const std::string message);
    bool key_is_deprecated(std::string full_key) const;
    bool key_is_renamed(std::string full_key);
    void raise_key_is_rename_error(std::string full_key);
    bool is_new_allowed() const;
    bool load_cfg(std::string cfg_filename);
    bool load_cfg_from_yaml_str(std::string str_obj);


// class attributes:
    PathManager pathman;
    std::string name;
    // std::unordered_map< std::string, CfgNode* > contents;
    std::unordered_map< std::string, std::unordered_map<std::string, float> > contents;//need this setup for configuring the model in MaskDINO2.cpp
    std::vector<CfgNode*> children;
    std::vector<std::string> key_list;
};

// CODE FROM:
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/config.py#L87
void CfgNode::merge_from_file(std::string cfg_filename, bool allow_unsafe = true) {
    if (!(pathman.isfile(cfg_filename))) {
        std::cout << "Config file " << cfg_filename << " does not exist." << std::endl;
    }
    CfgNode loaded_cfg = load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe);
}

CfgNode CfgNode::load_yaml_with_base(std::string filename, bool allow_unsafe) {
    open_cfg(filename);
    try {
        CfgNode cfg;
    }
    catch(...) {
        ;// https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/config.py
    }
    CfgNode tmp;
    return tmp;
}

CfgNode CfgNode::merge_a_into_b(CfgNode a, CfgNode b) {
    return *this;
}

CfgNode CfgNode::load_with_base(std::string cfg_base_filename) {
    return *this;
}

void CfgNode::dump() { //idk if i need this one...
    ;
}

CfgNode CfgNode::merge_from_other_cfg(CfgNode cfg_other) {
    return *this;
}

// IDK WHAT TO PUT INSIDE VECTOR PARAMETER BELOW!!!
CfgNode CfgNode::merge_from_vector(std::vector<std::string> cfg_vector) { // adaptation from original method: merge_from_list(List cfg_list)
    return *this;
}

void CfgNode::setattr(const std::string name, const std::string val) {
    ;
}

CfgNode CfgNode::create_config_tree_from_dict(const std::unordered_map<std::string, std::string> dic, //intentionally not a referece! need a copy...
                                              const std::vector<std::string> &key_list) {
    CfgNode ret;
    for ( const auto & [ key, value ] : dic )
    {
        if (value) {
            key_list.push_back(key);
            dic[key] = CfgNode(value, key_list);
        }
        else {

        }
    }
    return ret;
}

std::vector<std::string> CfgNode::getattr(const std::string name) const {
    return std::vector<std::string>{name};
}

void CfgNode::str() const {
    ;
}

void CfgNode::repr() const {
    ;
}

void CfgNode::freeze() {
    ;
}

void CfgNode::defrost() {
    ;
}

bool CfgNode::is_frozen() const {
    return false;
}

void CfgNode::immutable(bool is_immutable) {
    ;
}

CfgNode CfgNode::clone() const {
    return *this;
}

void CfgNode::register_deprecated_key(const std::string old_name,
                                      const std::string new_name,
                                      const std::string message) {
    ;
}

bool CfgNode::key_is_deprecated(std::string full_key) const {
    return false;
}
bool CfgNode::key_is_renamed(std::string full_key) {
    return false;
}
void CfgNode::raise_key_is_rename_error(std::string full_key) {
    ;
}

bool CfgNode::is_new_allowed() const {
    return false;
}

bool CfgNode::load_cfg(std::string cfg_filename) {
    return false;
}

bool CfgNode::load_cfg_from_yaml_str(std::string str_obj) {
    return false;
}

#endif