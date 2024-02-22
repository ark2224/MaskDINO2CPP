#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef PATHMANAGER_H
#define PATHMANAGER_H

// https://github.com/facebookresearch/iopath/blob/main/iopath/common/file_io.py
// https://github.com/facebookresearch/fairseq/blob/main/fairseq/file_io.py
class PathManager {
public:
    bool isfile(const std::string);
    void open(const std::string, const std::string);
    void get_supported_prefixes() { std::logic_error("PathManager not implemented correctly.");  }
    std::string get_local_path(const std::string path, bool force = false);
    bool copy_from_local(const std::string local_path, const std::string dst_path, bool overwrite = false);
    void opent(const std::string path, const std::string mode = "r", const int buffering = 32);
    void opena(const std::string path, const std::string mode = "r", const int buffering = -1);
    bool async_join(const std::string);


private:
    bool non_blocking_io_manager = false;
};

bool PathManager::isfile(const std::string filename) {
    return false;
}

void PathManager::open(const std::string filename, const std::string opt = "r") {
    ;
}

bool PathManager::async_join(const std::string path = "") {
    if (!non_blocking_io_manager) {
        
    }
    return false;
}

#endif