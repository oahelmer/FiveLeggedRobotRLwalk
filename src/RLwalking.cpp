#include "RLwalking.h"
#include <stdexcept>
#include <iostream>

RLController::RLController(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "RLController"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    try {
        session = Ort::Session(env, model_path.c_str(), opts);
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load rl_policy.onnx");
    }
}

std::vector<float> RLController::compute_action(const std::vector<float>& obs) {
    if (obs.size() != 17)
        throw std::invalid_argument("Observation must be 17-dimensional");

    std::vector<int64_t> shape = {1, 17};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(obs.data()),
        obs.size(), shape.data(), shape.size());

    const char* in[]  = {"obs"};
    const char* out[] = {"action"};

    auto result = session.Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);

    float* ptr = result.front().GetTensorMutableData<float>();
    return std::vector<float>(ptr, ptr + 5);   // 5-dimensional action
}
