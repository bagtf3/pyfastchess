// batcher.cpp
#include "batcher.hpp"
#include <iostream>
#include <cassert>

namespace py = pybind11;

Batcher& Batcher::instance() {
    static Batcher inst;
    return inst;
}

Batcher::Batcher() {
    // default session options can be tuned later (CUDA provider etc.)
    session_options_.SetIntraOpNumThreads(1);
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
    session_.reset(new Ort::Session(env_, std::wstring(model_path_.begin(), model_path_.end()).c_str(), session_options_));
    session_configured_.store(true);
}

void Batcher::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return; // already started
    running_ = true;
    worker_ = std::thread([this]{ this->worker_loop(); });
}

void Batcher::stop() {
    {
        std::lock_guard<std::mutex> l(mu_);
        running_ = false;
        cv_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    started_.store(false);
}

void Batcher::push_prediction(uint64_t token, uint64_t zobrist, py::array_t<float> input_arr) {
    // validate input dtype/ndim in Python before calling to avoid costly checks; still check shape minimally
    py::buffer_info bi = input_arr.request();
    if (bi.ndim < 1) throw std::runtime_error("Batcher: input must be an array (NHWC / flattened)");
    std::lock_guard<std::mutex> l(mu_);
    queue_.push_back(Item{token, zobrist, input_arr});
    if (queue_.size() >= queue_size_) cv_.notify_one();
}

void Batcher::force_predict() {
    std::lock_guard<std::mutex> l(mu_);
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

void Batcher::worker_loop() {
    while (true) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]{ return !running_ || queue_.size() >= queue_size_ || (!queue_.empty()); });
        if (!running_ && queue_.empty()) break;

        // swap out queued items for processing
        std::vector<Item> items;
        items.swap(queue_);
        lk.unlock();

        // build input batch lists
        std::vector<py::array_t<float>> inputs;
        std::vector<uint64_t> zkeys;
        std::vector<uint64_t> tokens;
        inputs.reserve(items.size());
        zkeys.reserve(items.size());
        tokens.reserve(items.size());
        for (auto &it : items) {
            inputs.push_back(it.arr);
            zkeys.push_back(it.zobrist);
            tokens.push_back(it.token);
        }

        // ensure session exists and run batch
        try {
            ensure_session();
            run_batch(zkeys, inputs, tokens);
        } catch (const std::exception &e) {
            std::cerr << "[Batcher] prediction error: " << e.what() << "\n";
            // store empty/failed marker if desired
        }
    }
}

void Batcher::run_batch(std::vector<uint64_t> const& zobrist_keys,
                        std::vector<py::array_t<float>> const& inputs,
                        std::vector<uint64_t> const& tokens) {
    // POC: assume each input has same shape and contiguous float32
    if (inputs.empty()) return;
    auto info0 = inputs[0].request();
    size_t per_item_elems = 1;
    for (int d=0; d<info0.ndim; ++d) per_item_elems *= static_cast<size_t>(info0.shape[d]);

    size_t B = inputs.size();
    std::vector<float> batch(B * per_item_elems);
    for (size_t i=0;i<B;++i) {
        auto bi = inputs[i].request();
        assert(bi.size == (ssize_t)per_item_elems);
        std::memcpy(batch.data() + i*per_item_elems, bi.ptr, per_item_elems * sizeof(float));
    }

    // --- Build ONNX input/outputs and run (minimal POC) ---
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_->GetInputNameAllocated(0, alloc);
    std::string in_s = in_name.get();
    const char* in_names[] = { in_s.c_str() };

    // input shape: [B, ...] -> build shape vector
    std::vector<int64_t> ishape;
    ishape.push_back(static_cast<int64_t>(B));
    for (int d=0; d<info0.ndim; ++d) ishape.push_back(static_cast<int64_t>(info0.shape[d]));

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, batch.data(), batch.size(), ishape.data(), ishape.size());

    // outputs
    size_t out_count = session_->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_hold;
    std::vector<std::string> out_s;
    std::vector<const char*> out_names;
    for (size_t i = 0; i < out_count; ++i) {
        out_name_hold.push_back(session_->GetOutputNameAllocated(i, alloc));
        out_s.emplace_back(out_name_hold.back().get());
        out_names.push_back(out_s.back().c_str());
    }

    auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names.data(), out_names.size());

    // Flatten outputs (POC: flatten all outputs sequentially into outputs_flat)
    // This is intentionally minimal — adapt to your actual output layout.
    std::vector<float> flat;
    for (size_t i=0;i<outs.size();++i) {
        auto ti = outs[i].GetTensorTypeAndShapeInfo();
        size_t cnt = ti.GetElementCount();
        const float* p = outs[i].GetTensorData<float>();
        flat.insert(flat.end(), p, p + cnt);
    }

    // Store per-item slices in results_ keyed by zobrist.
    std::lock_guard<std::mutex> l(mu_);
    for (size_t i=0;i<B;++i) {
        PredictionResult pr;
        // naive: store full flat buffer for now (consumer must know layout)
        pr.outputs_flat = flat;
        pr.shape = { static_cast<int64_t>(B) }; // placeholder
        results_[zobrist_keys[i]] = std::move(pr);
    }
}
