#include <iterator>
#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef REGISTRY_H
#define REGISTRY_H


// from here too: https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/registry.py
class Registry : std::vector<std::string> {
public:
// constructors:
    Registry(const std::string s) : name(s) { }
// class methods:
    void do_register(std::string name, torch::nn::Module obj);//unsure what the object should be (it's originally 'Any')
    torch::nn::Module register_(torch::nn::Module obj);
    torch::nn::Module *get(std::string obj);//made the return type POINTER to do dynamic_cast<Backbone> and dynamic_cast<SemanticSegmentor> in detectron2.h
    bool contains(std::string name);
    std::string repr();
    std::vector<std::string>::iterator iter();
    void doc(const std::string s) { documentation = s; }


private:
    std::string name;
    std::unordered_map<std::string, torch::nn::Module*> obj_map = std::unordered_map<std::string, torch::nn::Module*>();
    std::string documentation;
};

void Registry::do_register(std::string name, torch::nn::Module obj) { //unsure what the object should be (it's originally 'Any')
    ;
}

torch::nn::Module Registry::register_(torch::nn::Module obj) {
    return obj;
}

torch::nn::Module *Registry::get(std::string obj) {//made the return type POINTER to do dynamic_cast<Backbone> and dynamic_cast<SemanticSegmentor> in detectron2.h
    return (*this).obj_map.at(obj);
}

bool Registry::contains(std::string name) {
    return false;
}

std::string Registry::repr() {
    return "";
}

std::vector<std::string>::iterator Registry::iter() {
    return (*this).begin();
}





#endif