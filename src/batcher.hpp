// src/batcher.hpp
#pragma once

#include <onnxruntime_cxx_api.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct PredictionResult {
    std::vector<float> outputs_flat;    // flattened concatenation of all output tensors
    std::vector<int64_t> shape;         // placeholder shape info (model-specific)
};

class Batcher {
public:
    // singleton
    static Batcher& instance();

    // lifecycle
    void load_model(const std::string& onnx_path);
    void start();
    void stop();

    // tuning
    void set_queue_size(size_t q);

    // queueing: token can be 0 if unused, zobrist is required key, input is raw ready-to-run bytes/values
    // We keep the signature using std::vector<float> for model-ready float inputs (caller converts).
    void push_prediction(uint64_t token, uint64_t zobrist, const std::vector<float>& input);

    // force an immediate prediction run on whatever is queued
    void force_predict();

    // get result if available
    std::optional<PredictionResult> get_result(uint64_t zobrist);

    // clear results cache
    void clear_results_cache();

    // stats for Python (converted automatically by pybind11)
    std::map<std::string, long long> stats_map();

    ~Batcher();

private:
    Batcher();
    Batcher(const Batcher&) = delete;
    Batcher& operator=(const Batcher&) = delete;

    void worker_loop();
    void run_batch(const std::vector<uint64_t>& zobrist_keys,
                   const std::vector<std::vector<float>>& inputs,
                   const std::vector<uint64_t>& tokens);

    // ONNX helpers
    void ensure_session();

    // concurrency + queue
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::atomic<bool> started_{false};

    size_t queue_size_{8};
    struct Item { uint64_t token; uint64_t zobrist; std::vector<float> input; };
    std::vector<Item> queue_;

    // results
    std::unordered_map<uint64_t, PredictionResult> results_;

    // ONNX
    std::string model_path_;
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "batcher"};
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;
    std::atomic<bool> session_configured_{false};

    // basic stats
    std::atomic<long long> stat_total_batches_{0};
    std::atomic<long long> stat_total_requests_{0};
    std::atomic<long long> stat_total_predictions_{0};
    std::atomic<long long> stat_failed_predictions_{0};
    std::atomic<long long> stat_last_batch_size_{0};
};
