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

void Batcher::set_cpu_threads(int intra, int inter) {
    cpu_intra_threads_ = intra;
    cpu_inter_threads_ = inter;
}

std::vector<Batcher::BatchHistoryEntry> Batcher::batch_history() {
    std::lock_guard<std::mutex> lk(hist_mu_);
    return std::vector<Batcher::BatchHistoryEntry>(hist_.begin(), hist_.end());
}


void Batcher::clear_batch_history() {
    std::lock_guard<std::mutex> lk(hist_mu_);
    hist_.clear();
    hist_counter_ = 0;
}

void Batcher::load_model(const std::string& onnx_path) {
    // store path (if your start() uses it later)
    model_path_ = onnx_path;

    // destroy old session if any
    session_.reset();

    session_options_ = Ort::SessionOptions{};
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // CPU likes parallel; give it threads
    int intra = cpu_intra_threads_ > 0
              ? cpu_intra_threads_
              : (int)std::thread::hardware_concurrency();
    int inter = cpu_inter_threads_ > 0 ? cpu_inter_threads_ : 1;
    session_options_.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options_.SetIntraOpNumThreads(intra);
    session_options_.SetInterOpNumThreads(inter);

    // CPU EP is default — do NOT append CUDA here
#ifdef _WIN32
    // onnxruntime C++ API wants ORTCHAR_T* (wchar_t on Windows)
    std::wstring wpath(model_path_.begin(), model_path_.end());
    session_ = std::make_unique<Ort::Session>(
        ort_env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(
        ort_env_, model_path_.c_str(), session_options_);
#endif

    session_configured_.store(true, std::memory_order_release);

    // ---- detect input layout/dims from the model ----
    {
        Ort::AllocatorWithDefaultOptions alloc;
        if (session_->GetInputCount() != 1) {
            std::cerr << "[Batcher] WARN: expected 1 input, got "
                    << session_->GetInputCount() << std::endl;
        }
        Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
        auto tt = ti.GetTensorTypeAndShapeInfo();
        auto dims = tt.GetShape();  // may contain -1 for dynamic batch
        if (dims.size() != 4) {
            std::cerr << "[Batcher] WARN: unexpected input rank " << dims.size()
                    << " (expected 4)" << std::endl;
        } else {
            // dims is either [N,H,W,C] or [N,C,H,W]
            // Decide by which slot equals the channel count (70).
            // If dynamic, dims could be -1; we’ll fall back to heuristics.
            int64_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];

            // Heuristic: if d3==70 => NHWC; if d1==70 => NCHW; otherwise fall back to NHWC.
            if (d3 == 70) {
                input_is_nchw_ = false;
                in_C_ = 70; in_H_ = (d1 > 0 ? d1 : 8); in_W_ = (d2 > 0 ? d2 : 8);
            } else if (d1 == 70) {
                input_is_nchw_ = true;
                in_C_ = 70; in_H_ = (d2 > 0 ? d2 : 8); in_W_ = (d3 > 0 ? d3 : 8);
            } else {
                // fallback assume NHWC 8x8x70
                input_is_nchw_ = false;
                in_C_ = 70; in_H_ = 8; in_W_ = 8;
                std::cerr << "[Batcher] WARN: could not infer layout from dims; assuming NHWC."
                        << std::endl;
            }
            std::cout << "[Batcher] Input layout: " << (input_is_nchw_ ? "NCHW" : "NHWC")
                    << "  C=" << in_C_ << " H=" << in_H_ << " W=" << in_W_ << std::endl;
        }
    }

    std::cout << "[Batcher] CPU session created. intra=" << intra
              << " inter=" << inter
              << " inputs=" << session_->GetInputCount()
              << " outputs=" << session_->GetOutputCount()
              << std::endl;
}


void Batcher::ensure_session() {
    if (session_configured_.load()) return;
    if (model_path_.empty()) throw std::runtime_error("Batcher: model not set");

    // Configure session options
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // try SEQUENTIAL; change to PARALLEL if desired
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);

    // Try to attach CUDA provider using OrtCUDAProviderOptions (many ORT builds expect a struct)
    bool cuda_requested = false;
    try {
        ::OrtCUDAProviderOptions cuda_opts{}; // zero-init and set device id
        cuda_opts.device_id = 0;
        // This signature exists in many ORT C++ wrappers: AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions &)
        opts.AppendExecutionProvider_CUDA(cuda_opts);
        cuda_requested = true;
        std::cerr << "[Batcher] requested CUDA execution provider (device 0)\n";
    } catch (const std::exception &e) {
        std::cerr << "[Batcher] AppendExecutionProvider_CUDA threw: " << e.what() << "\n";
        std::cerr << "[Batcher] continuing without explicitly requesting CUDA provider\n";
    } catch (...) {
        std::cerr << "[Batcher] AppendExecutionProvider_CUDA not available / failed (unknown error)\n";
    }

    // Create session (handle Windows wide-string path)
#ifdef _WIN32
    std::wstring wpath = to_wstring(model_path_);
    session_.reset(new Ort::Session(ort_env_, wpath.c_str(), opts));
#else
    session_.reset(new Ort::Session(ort_env_, model_path_.c_str(), opts));
#endif

    // Informational note: we requested CUDA above if available.
    if (!cuda_requested) {
        std::cerr << "[Batcher] note: CUDA provider was not requested at session creation.\n";
        std::cerr << "[Batcher] If you expect GPU execution, ensure you're linking a GPU-enabled ONNX Runtime\n";
        std::cerr << "[Batcher] (e.g. the 'onnxruntime-win-x64-gpu' package) and that AppendExecutionProvider_CUDA\n";
        std::cerr << "[Batcher] is available in your ORT headers/binary.\n";
    } else {
        std::cerr << "[Batcher] session created (CUDA provider was requested). Use `nvidia-smi` to verify GPU memory / utilization.\n";
    }

    // Warm-up: 1-2 dummy runs to JIT kernels / allocate device memory
    try {
        const int64_t H = 8, W = 8, C = 70; // model input expected HxWxC (POC)
        std::vector<int64_t> warm_shape = {1, H, W, C};
        std::vector<float> warm_buf(static_cast<size_t>(H * W * C), 0.0f);

        Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value warm_input = Ort::Value::CreateTensor<float>(meminfo, warm_buf.data(), warm_buf.size(),
                                                                warm_shape.data(), warm_shape.size());

        // keep allocated names alive while building out_names
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name_ptr = session_->GetInputNameAllocated(0, alloc);
        const char* in_names[] = { in_name_ptr.get() };

        size_t out_count = session_->GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> out_name_hold;
        std::vector<const char*> out_names;
        out_name_hold.reserve(out_count);
        out_names.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            out_name_hold.push_back(session_->GetOutputNameAllocated(i, alloc));
            out_names.push_back(out_name_hold.back().get());
        }

        Ort::RunOptions run_opts;
        for (int i = 0; i < 2; ++i) {
            auto outs = session_->Run(run_opts, in_names, &warm_input, 1, out_names.data(), out_names.size());
            (void)outs;
        }
        std::cerr << "[Batcher] warmup done\n";
    } catch (const std::exception &e) {
        std::cerr << "[Batcher] warmup error: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[Batcher] warmup unknown error\n";
    }

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
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const size_t B = inputs.size();
    if (B == 0) return;

    // ---- detect input layout & dims from the model (NHWC vs NCHW) ----
    // We assume channels == 70; dims may have -1 for N (dynamic batch).
    bool model_expects_nchw = false;
    int64_t C = 70, H = 8, W = 8;
    {
        Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
        auto tt = ti.GetTensorTypeAndShapeInfo();
        auto dims = tt.GetShape(); // rank should be 4
        if (dims.size() == 4) {
            const int64_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
            if (d3 == 70) {                // [N,H,W,C]
                model_expects_nchw = false;
                H = (d1 > 0 ? d1 : 8);
                W = (d2 > 0 ? d2 : 8);
                C = 70;
            } else if (d1 == 70) {         // [N,C,H,W]
                model_expects_nchw = true;
                H = (d2 > 0 ? d2 : 8);
                W = (d3 > 0 ? d3 : 8);
                C = 70;
            } else {
                // fallback assume NHWC
                model_expects_nchw = false;
                C = 70; H = 8; W = 8;
            }
        } else {
            // fallback assume NHWC [N,8,8,70]
            model_expects_nchw = false;
            C = 70; H = 8; W = 8;
        }
    }

    const size_t per_item = static_cast<size_t>(C * H * W);
    std::vector<float> buf;
    buf.resize(B * per_item);

    Ort::Value in0{nullptr};
    Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    if (!model_expects_nchw) {
        // ---- model expects NHWC: [B, H, W, C] ----
        // Your encoder already produces NHWC (H*W*C), so we can memcpy per sample.
        for (size_t i = 0; i < B; ++i) {
            const auto& src = inputs[i];
            if (src.size() != per_item) {
                std::cerr << "[Batcher] WARN: NHWC input size mismatch for item "
                          << i << " got " << src.size()
                          << " expected " << per_item << std::endl;
            }
            std::memcpy(buf.data() + i * per_item, src.data(),
                        std::min(src.size(), per_item) * sizeof(float));
        }
        std::array<int64_t, 4> shape{ static_cast<int64_t>(B), H, W, C };
        in0 = Ort::Value::CreateTensor<float>(
            meminfo, buf.data(), buf.size(), shape.data(), shape.size());
    } else {
        // ---- model expects NCHW: [B, C, H, W] ----
        // Transpose NHWC source -> NCHW destination while copying.
        for (size_t i = 0; i < B; ++i) {
            const auto& src = inputs[i];          // NHWC packed: (h*W + w)*C + c
            float* dst_base = buf.data() + i * per_item; // NCHW: c*(H*W) + (h*W + w)
            if (src.size() != per_item) {
                std::cerr << "[Batcher] WARN: NCHW input size mismatch for item "
                          << i << " got " << src.size()
                          << " expected " << per_item << std::endl;
            }
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    const size_t hw = static_cast<size_t>(h * W + w);
                    const size_t src_hw_off = hw * static_cast<size_t>(C);
                    const size_t dst_hw_off = hw; // added to c*(H*W)
                    for (int64_t c = 0; c < C; ++c) {
                        const size_t src_idx = src_hw_off + static_cast<size_t>(c);
                        const size_t dst_idx = static_cast<size_t>(c) * static_cast<size_t>(H * W) + dst_hw_off;
                        dst_base[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        std::array<int64_t, 4> shape{ static_cast<int64_t>(B), C, H, W };
        in0 = Ort::Value::CreateTensor<float>(
            meminfo, buf.data(), buf.size(), shape.data(), shape.size());
    }

    // ---- names: input is literally "board"; outputs queried safely ----
    const char* const in_names[1] = {"board"};

    // Keep AllocatedStringPtr alive while calling Run()
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t out_count = session_->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_storage;
    out_name_storage.reserve(out_count);
    std::vector<const char*> out_names;
    out_names.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        out_name_storage.emplace_back(session_->GetOutputNameAllocated(i, alloc));
        out_names.push_back(out_name_storage.back().get());
    }

    // ---- Run() ----
    const auto t_run0 = clock::now();
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 in_names, &in0, 1,
                                 out_names.data(), out_count);
    const auto t_run1 = clock::now();

    // ---- flatten outputs, stitch per-sample, and fill results_ ----
    // Expect 5 heads: value(B,1), from(B,64), to(B,64), piece(B,6), promo(B,4)
    std::vector<std::vector<float>> outs_flat;
    outs_flat.reserve(outputs.size());

    for (size_t j = 0; j < outputs.size(); ++j) {
        const float* p = outputs[j].GetTensorMutableData<float>();
        const auto& info = outputs[j].GetTensorTypeAndShapeInfo();
        const size_t n = info.GetElementCount();
        std::vector<float> tmp(n);
        std::memcpy(tmp.data(), p, n * sizeof(float));
        outs_flat.emplace_back(std::move(tmp));
    }

    {
        // guard writes to results_
        std::lock_guard<std::mutex> lk(mu_);

        // stitch row-by-row into PredictionResult expected by MCTS
        for (size_t i = 0; i < B; ++i) {
            std::vector<float> flat;
            flat.reserve(1 + 64 + 64 + 6 + 4);

            // value head (B,1) — often flattened to length B
            if (!outs_flat.empty()) {
                flat.push_back(outs_flat[0][i]);
            }
            // from (B,64)
            if (outs_flat.size() >= 2) {
                flat.insert(flat.end(),
                            outs_flat[1].begin() + i*64,
                            outs_flat[1].begin() + (i+1)*64);
            }
            // to (B,64)
            if (outs_flat.size() >= 3) {
                flat.insert(flat.end(),
                            outs_flat[2].begin() + i*64,
                            outs_flat[2].begin() + (i+1)*64);
            }
            // piece (B,6)
            if (outs_flat.size() >= 4) {
                flat.insert(flat.end(),
                            outs_flat[3].begin() + i*6,
                            outs_flat[3].begin() + (i+1)*6);
            }
            // promo (B,4)
            if (outs_flat.size() >= 5) {
                flat.insert(flat.end(),
                            outs_flat[4].begin() + i*4,
                            outs_flat[4].begin() + (i+1)*4);
            }

            PredictionResult pr;
            pr.outputs_flat = std::move(flat);
            pr.shape = {1, 64, 64, 6, 4};
            results_[zobrist_keys[i]] = std::move(pr);
        }
    }

    const auto t1 = clock::now();
    const long long run_ms =
        (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_run1 - t_run0).count();
    const long long total_ms =
        (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    stat_total_batches_++;
    stat_last_batch_size_ = (long long)B;
    stat_total_predictions_ += (long long)B;

    // record timing history
    {
        std::lock_guard<std::mutex> lk2(hist_mu_);
        BatchHistoryEntry e;
        e.id = ++hist_counter_;
        e.size = B;
        e.run_ms = run_ms;
        e.total_ms = total_ms;
        e.t_start_ns = now_ns();
        e.t_end_ns   = now_ns();
        hist_.push_back(e);
        if (hist_.size() > hist_cap_) hist_.pop_front();
    }
}