// src/batcher.cpp
#include "batcher.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring> // memcpy

namespace {

// minimal portable utf8 -> wstring conversion for local POC
static std::wstring to_wstring(const std::string& s) {
    // Portable (basic) conversion: OK for local paths on Windows for POC.
    return std::wstring(s.begin(), s.end());
}

} // anon

Batcher& Batcher::instance() {
    static Batcher inst;
    return inst;
}

Batcher::Batcher() {
    session_options_.SetIntraOpNumThreads(1);
    // Do not configure providers here — load_model() can set options later if needed.
}

Batcher::~Batcher() {
    stop();
}

void Batcher::load_model(const std::string& onnx_path) {
    std::lock_guard<std::mutex> l(mu_);
    model_path_ = onnx_path;
    session_.reset();
    session_configured_.store(false);
}

void Batcher::ensure_session() {
    if (session_configured_.load()) return;
    if (model_path_.empty()) throw std::runtime_error("Batcher: model not set");

#ifdef _WIN32
    std::wstring wpath = to_wstring(model_path_);
    session_.reset(new Ort::Session(ort_env_, wpath.c_str(), session_options_));
#else
    session_.reset(new Ort::Session(ort_env_, model_path_.c_str(), session_options_));
#endif

    session_configured_.store(true);
}

void Batcher::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    running_.store(true);
    worker_ = std::thread([this]{ this->worker_loop(); });
}

void Batcher::stop() {
    {
        std::lock_guard<std::mutex> l(mu_);
        running_.store(false);
        cv_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    started_.store(false);
}

void Batcher::set_queue_size(size_t q) {
    std::lock_guard<std::mutex> l(mu_);
    queue_size_ = (q == 0) ? 1 : q;
}

void Batcher::push_prediction(uint64_t token, uint64_t zobrist, const std::vector<float>& input) {
    {
        std::lock_guard<std::mutex> l(mu_);
        queue_.push_back(Item{token, zobrist, input});
        stat_total_requests_.fetch_add(1);
    }
    if (queue_.size() >= queue_size_) cv_.notify_one();
}

void Batcher::force_predict() {
    cv_.notify_one();
}

std::optional<PredictionResult> Batcher::get_result(uint64_t zobrist) {
    std::lock_guard<std::mutex> l(mu_);
    auto it = results_.find(zobrist);
    if (it == results_.end()) return std::nullopt;
    return it->second;
}

void Batcher::clear_results_cache() {
    std::lock_guard<std::mutex> l(mu_);
    results_.clear();
}

std::map<std::string, long long> Batcher::stats_map() {
    std::map<std::string, long long> m;
    m["total_batches"] = stat_total_batches_.load();
    m["total_requests"] = stat_total_requests_.load();
    m["total_predictions"] = stat_total_predictions_.load();
    m["failed_predictions"] = stat_failed_predictions_.load();
    m["last_batch_size"] = stat_last_batch_size_.load();
    return m;
}

void Batcher::worker_loop() {
    while (true) {
        std::vector<Item> items;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return !running_.load() || queue_.size() >= queue_size_ || !queue_.empty(); });
            if (!running_.load() && queue_.empty()) break;
            items.swap(queue_);
        }

        if (items.empty()) continue;

        std::vector<std::vector<float>> inputs;
        std::vector<uint64_t> zobrist_keys;
        std::vector<uint64_t> tokens;
        inputs.reserve(items.size()); zobrist_keys.reserve(items.size()); tokens.reserve(items.size());
        for (auto &it : items) {
            inputs.push_back(it.input);
            zobrist_keys.push_back(it.zobrist);
            tokens.push_back(it.token);
        }

        try {
            ensure_session();
            run_batch(zobrist_keys, inputs, tokens);
            stat_total_batches_.fetch_add(1);
        } catch (const std::exception &e) {
            std::cerr << "[Batcher] prediction error: " << e.what() << "\n";
            stat_failed_predictions_.fetch_add(1);
        }
    }
}

void Batcher::run_batch(const std::vector<uint64_t>& zobrist_keys,
                        const std::vector<std::vector<float>>& inputs,
                        const std::vector<uint64_t>& tokens) {
    if (!session_) throw std::runtime_error("Batcher: session not configured");

    size_t B = inputs.size();
    if (B == 0) return;
    // assume all inputs have same length
    size_t per_item = inputs[0].size();
    for (const auto &v : inputs) if (v.size() != per_item)
        throw std::runtime_error("Batcher: inconsistent input lengths");

    // build batch buffer
    std::vector<float> batch; batch.resize(B * per_item);
    for (size_t i=0;i<B;++i) std::memcpy(batch.data() + i*per_item, inputs[i].data(), per_item * sizeof(float));

    // prepare ONNX input
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr = session_->GetInputNameAllocated(0, alloc);
    std::string in_name = in_name_ptr.get();
    const char* in_names[] = { in_name.c_str() };

    // build shape: [B, ...] assuming session input dims known from first input shape
    // For POC, assume model expects [B,8,8,channels] flattened as per_item => we pass [B, per_item] as 2D
    std::vector<int64_t> input_shape = { static_cast<int64_t>(B), static_cast<int64_t>(per_item) };

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, batch.data(), batch.size(), input_shape.data(), input_shape.size());

    // gather output names safely
    size_t out_count = session_->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_hold;
    std::vector<const char*> out_names;
    for (size_t i=0;i<out_count;++i) {
        out_name_hold.push_back(session_->GetOutputNameAllocated(i, alloc));
        const char* p = out_name_hold.back().get();
        if (!p || p[0] == '\0') throw std::runtime_error("Batcher: model has empty output name");
        out_names.push_back(p);
    }

    // run
    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names.data(), out_names.size());

    // flatten all outputs sequentially
    std::vector<float> flat;
    for (size_t oi=0; oi<outs.size(); ++oi) {
        auto &o = outs[oi];
        auto ti = o.GetTensorTypeAndShapeInfo();
        size_t cnt = ti.GetElementCount();
        const float* p = o.GetTensorData<float>();
        flat.insert(flat.end(), p, p + cnt);
    }

    // store results per-item (for now, store same flat vector and shape; consumer decodes)
    {
        std::lock_guard<std::mutex> l(mu_);
        stat_total_predictions_.fetch_add(static_cast<long long>(B));
        stat_last_batch_size_.store(static_cast<long long>(B));
        for (size_t i=0;i<B;++i) {
            PredictionResult pr;
            pr.outputs_flat = flat;                 // naive: consumer must decode per-item from layout
            pr.shape = { static_cast<int64_t>(B) }; // placeholder; can be extended with full output shapes
            results_[zobrist_keys[i]] = std::move(pr);
        }
    }
}
