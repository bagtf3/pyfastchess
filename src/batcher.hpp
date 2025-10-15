#pragma once
#include <onnxruntime_cxx_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <optional>
#include <string>
#include <atomic>

struct PredictionResult {
    // simple POC: vector of floats for value/head + vector of head logits/prior-like arrays
    std::vector<float> outputs_flat; // raw outputs flattened (caller decodes)
    std::vector<int64_t> shape;      // shape of the first output (e.g. {B, H, ...}) or head lengths
};

class Batcher {
public:
    static Batcher& instance(); // singleton

    // lifecycle
    void load_model(const std::string& onnx_path);
    void start();   // spawn worker
    void stop();    // stop worker and join

    // tuning
    void set_queue_size(size_t q) { std::lock_guard<std::mutex> l(mu_); queue_size_ = q; }

    // push: token is optional (you can pass 0), zobrist is required key, input_arr is float32 numpy already prepared
    void push_prediction(uint64_t token, uint64_t zobrist, pybind11::array_t<float> input_arr);

    // force an immediate prediction on whatever is queued
    void force_predict();

    // get result if available
    std::optional<PredictionResult> get_result(uint64_t zobrist);

    // clear results cache
    void clear_results_cache();

    ~Batcher();

private:
    Batcher(); // private for singleton

    // internal run loop
    void worker_loop();
    void run_batch(std::vector<uint64_t> const& zobrist_keys,
                   std::vector<pybind11::array_t<float>> const& inputs,
                   std::vector<uint64_t> const& tokens);

    // ONNX helpers (POC)
    void ensure_session();

    // data
    std::mutex mu_;
    std::condition_variable cv_;
    bool running_{false};
    std::thread worker_;
    size_t queue_size_{8};

    // queued items
    struct Item { uint64_t token; uint64_t zobrist; pybind11::array_t<float> arr; };
    std::vector<Item> queue_;

    // results cache
    std::unordered_map<uint64_t, PredictionResult> results_;

    // ONNX
    std::string model_path_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "batcher"};
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    std::atomic<bool> session_configured_{false};

    // minimal guard to avoid double-start
    std::atomic<bool> started_{false};
};
