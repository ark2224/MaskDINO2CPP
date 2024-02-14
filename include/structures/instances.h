#include <array>
#include <iostream>
#include <torch/torch.h>
#include <string>
#include <vector>
#include "boxes.h"

#ifndef INSTANCES_H
#define INSTANCES_H

class Instances : std::unordered_map<std::string, Boxes> {
public:
    int (*get_image_size())[3] { return &image_size; } 
    void setattr(const std::string &name, const Boxes &value) { (*this)[name] = value; } //name is not used!! should I be using an unordered_map instead of a vector?
    void getattr(const std::string &name, const Boxes &value);
    void set(const std::string &name, const Boxes &value);
    bool has(const std::string &name) const { return ((*this).find(name) != (*this).end()); }
    void remove(const std::string &name) { (*this).erase(name); }
    Boxes get(const std::string &name) const;// { return (*this).at(name); }
    Instances to();
    Instances getitem();//????????? dont know what to put into the parameters
    size_t len() const { return (*this).bucket_count(); }
    iterator iter() { return (*this).begin(); }
    static Instances cat(std::vector<Instances>);
    std::string str();

private:
    int image_size[3];
};


#endif