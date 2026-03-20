#ifndef RLWALKING_H
#define RLWALKING_H

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

class RLController {
private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info;

public:
    RLController(const std::string& model_path);
    
    // Takes 22-dimensional observation, returns 15-dimensional action
    std::vector<float> compute_action(const std::vector<float>& obs);
};

#endif