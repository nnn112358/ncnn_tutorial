# 8. パフォーマンス最適化

ncnnアプリケーションの性能を最大限に引き出すためには、適切な最適化技術を適用する必要があります。本章では、マルチスレッド、GPU加速、メモリ最適化、プロファイリングなど、実践的な最適化手法を詳しく解説します。

## 8.1 マルチスレッド設定

ncnnは、効率的なマルチスレッド処理をサポートしており、適切な設定により大幅な性能向上が期待できます。

### 基本的なスレッド設定

```cpp
#include <ncnn/net.h>
#include <thread>

class ThreadOptimizedInference {
private:
    ncnn::Net net;
    int optimal_threads;

public:
    ThreadOptimizedInference() {
        // システムのCPUコア数を取得
        int cpu_cores = std::thread::hardware_concurrency();

        // 効率的なスレッド数の決定
        optimal_threads = determine_optimal_threads(cpu_cores);

        // ncnnのスレッド設定
        net.opt.num_threads = optimal_threads;
        net.opt.use_packing_layout = true;    // パッキングレイアウトを有効
        net.opt.use_winograd_convolution = true;  // Winograd畳み込みを有効
        net.opt.use_sgemm_convolution = true;     // SGEMM畳み込みを有効

        std::cout << "Using " << optimal_threads << " threads for inference" << std::endl;
    }

private:
    int determine_optimal_threads(int cpu_cores) {
        // CPU型別の最適化
        if (cpu_cores <= 2) {
            return cpu_cores;  // 低性能CPUでは全コアを使用
        } else if (cpu_cores <= 4) {
            return cpu_cores - 1;  // 1コアをシステム用に残す
        } else if (cpu_cores <= 8) {
            return std::min(4, cpu_cores - 1);  // 最大4スレッド
        } else {
            return std::min(6, cpu_cores / 2);  // 高性能CPUでは半分程度
        }
    }
};
```

### 動的スレッド調整

```cpp
class AdaptiveThreadManager {
private:
    ncnn::Net* net;
    int min_threads;
    int max_threads;
    std::vector<double> performance_history;

public:
    AdaptiveThreadManager(ncnn::Net* network, int min_t = 1, int max_t = 8)
        : net(network), min_threads(min_t), max_threads(max_t) {}

    void optimize_thread_count(const std::function<void()>& inference_func) {
        int best_threads = min_threads;
        double best_time = std::numeric_limits<double>::max();

        for (int threads = min_threads; threads <= max_threads; threads++) {
            net->opt.num_threads = threads;

            // ウォームアップ実行
            inference_func();

            // ベンチマーク実行
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; i++) {
                inference_func();
            }
            auto end = std::chrono::high_resolution_clock::now();

            double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;

            std::cout << "Threads: " << threads << ", Avg time: " << avg_time << " ms" << std::endl;

            if (avg_time < best_time) {
                best_time = avg_time;
                best_threads = threads;
            }
        }

        net->opt.num_threads = best_threads;
        std::cout << "Optimal thread count: " << best_threads << std::endl;
    }
};
```

### スレッドプールの実装

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

class InferenceThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    InferenceThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return res;
    }

    ~InferenceThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread &worker : workers) {
            worker.join();
        }
    }
};
```

## 8.2 Vulkanバックエンドの活用

VulkanはncnnでサポートされているGPU加速技術で、適切に設定することで大幅な性能向上が可能です。

### Vulkan設定の基本

```cpp
#include <ncnn/gpu.h>

class VulkanOptimizedInference {
private:
    ncnn::Net net;
    bool vulkan_available;

public:
    VulkanOptimizedInference() {
        // Vulkan対応の確認
        vulkan_available = ncnn::get_gpu_count() > 0;

        if (vulkan_available) {
            std::cout << "Vulkan GPUs available: " << ncnn::get_gpu_count() << std::endl;

            // GPU情報の表示
            for (int i = 0; i < ncnn::get_gpu_count(); i++) {
                const ncnn::GpuInfo& gpu_info = ncnn::get_gpu_info(i);
                std::cout << "GPU " << i << ": " << gpu_info.device_name()
                         << " (Compute capability: " << gpu_info.api_version() << ")" << std::endl;
            }

            setup_vulkan_optimizations();
        } else {
            std::cout << "Vulkan not available, using CPU" << std::endl;
            setup_cpu_optimizations();
        }
    }

private:
    void setup_vulkan_optimizations() {
        // Vulkan使用の有効化
        net.opt.use_vulkan_compute = true;

        // 最適なGPUデバイスの選択（通常は0番）
        net.opt.vulkan_device = 0;

        // Vulkan固有の最適化設定
        net.opt.use_fp16_packed = true;        // FP16パッキングを有効
        net.opt.use_fp16_storage = true;       // FP16ストレージを有効
        net.opt.use_fp16_arithmetic = true;    // FP16演算を有効
        net.opt.use_int8_storage = true;       // INT8ストレージを有効

        // メモリ最適化
        net.opt.use_image_storage = true;      // イメージストレージを使用
        net.opt.use_tensor_storage = true;     // テンソルストレージを使用
    }

    void setup_cpu_optimizations() {
        // CPU最適化設定
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = std::thread::hardware_concurrency();
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.use_int8_inference = true;
    }
};
```

### Vulkanメモリ最適化

```cpp
class VulkanMemoryOptimizer {
public:
    static void optimize_vulkan_memory(ncnn::Net& net) {
        // Vulkanワークスペースサイズの最適化
        net.opt.workspace_vkdev = ncnn::get_gpu_device(0);

        // メモリプールサイズの設定
        size_t available_memory = get_gpu_memory_size();
        size_t optimal_workspace = calculate_optimal_workspace(available_memory);

        net.opt.workspace_size_mb = optimal_workspace / (1024 * 1024);

        std::cout << "Vulkan workspace size: " << net.opt.workspace_size_mb << " MB" << std::endl;
    }

private:
    static size_t get_gpu_memory_size() {
        // GPUメモリサイズの取得（簡略化された例）
        const ncnn::GpuInfo& gpu_info = ncnn::get_gpu_info(0);
        // 実際の実装では、Vulkanドライバーからメモリ情報を取得
        return 4ULL * 1024 * 1024 * 1024;  // 4GB と仮定
    }

    static size_t calculate_optimal_workspace(size_t total_memory) {
        // 利用可能メモリの50-70%をワークスペースとして使用
        return static_cast<size_t>(total_memory * 0.6);
    }
};
```

### GPU/CPU ハイブリッド実行

```cpp
class HybridInference {
private:
    ncnn::Net gpu_net;
    ncnn::Net cpu_net;
    bool use_hybrid;

public:
    HybridInference() {
        use_hybrid = ncnn::get_gpu_count() > 0;

        if (use_hybrid) {
            // GPU用ネットワークの設定
            gpu_net.opt.use_vulkan_compute = true;
            gpu_net.opt.vulkan_device = 0;

            // CPU用ネットワークの設定
            cpu_net.opt.use_vulkan_compute = false;
            cpu_net.opt.num_threads = 2;  // GPU使用時はCPUスレッド数を減らす
        }
    }

    ncnn::Mat inference_adaptive(const ncnn::Mat& input, const std::string& model_complexity) {
        if (!use_hybrid) {
            return inference_cpu(input);
        }

        // モデルの複雑さに基づいて実行環境を選択
        if (should_use_gpu(model_complexity, input)) {
            return inference_gpu(input);
        } else {
            return inference_cpu(input);
        }
    }

private:
    bool should_use_gpu(const std::string& complexity, const ncnn::Mat& input) {
        // 判断基準：
        // 1. 入力サイズが大きい
        // 2. モデルが複雑
        // 3. バッチサイズが大きい

        size_t input_size = input.w * input.h * input.c;
        bool large_input = input_size > 224 * 224 * 3;
        bool complex_model = complexity == "heavy" || complexity == "complex";

        return large_input || complex_model;
    }

    ncnn::Mat inference_gpu(const ncnn::Mat& input) {
        ncnn::Extractor ex = gpu_net.create_extractor();
        ex.input("input", input);

        ncnn::Mat output;
        ex.extract("output", output);
        return output;
    }

    ncnn::Mat inference_cpu(const ncnn::Mat& input) {
        ncnn::Extractor ex = cpu_net.create_extractor();
        ex.input("input", input);

        ncnn::Mat output;
        ex.extract("output", output);
        return output;
    }
};
```

## 8.3 メモリプールの最適化

効率的なメモリ管理は、ncnnアプリケーションの性能に大きく影響します。

### カスタムメモリアロケーター

```cpp
#include <ncnn/allocator.h>

class OptimizedAllocator : public ncnn::Allocator {
private:
    size_t total_allocated;
    size_t peak_usage;
    std::mutex alloc_mutex;

public:
    OptimizedAllocator() : total_allocated(0), peak_usage(0) {}

    virtual void* fastMalloc(size_t size) override {
        std::lock_guard<std::mutex> lock(alloc_mutex);

        void* ptr = ncnn::fastMalloc(size);
        if (ptr) {
            total_allocated += size;
            peak_usage = std::max(peak_usage, total_allocated);
        }

        return ptr;
    }

    virtual void fastFree(void* ptr) override {
        if (ptr) {
            std::lock_guard<std::mutex> lock(alloc_mutex);
            // サイズ情報の追跡（実際の実装では、ポインターからサイズを逆算）
            ncnn::fastFree(ptr);
        }
    }

    size_t get_peak_usage() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(alloc_mutex));
        return peak_usage;
    }

    void print_memory_stats() const {
        std::cout << "Peak memory usage: " << peak_usage / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "Current allocated: " << total_allocated / (1024.0 * 1024.0) << " MB" << std::endl;
    }
};

class MemoryOptimizedNet {
private:
    ncnn::Net net;
    OptimizedAllocator* custom_allocator;

public:
    MemoryOptimizedNet() {
        custom_allocator = new OptimizedAllocator();

        // カスタムアロケーターの設定
        net.opt.blob_allocator = custom_allocator;
        net.opt.workspace_allocator = custom_allocator;

        // メモリ最適化オプション
        net.opt.use_memory_pool = true;
        net.opt.workspace_size_mb = 256;  // 256MBのワークスペース
    }

    ~MemoryOptimizedNet() {
        custom_allocator->print_memory_stats();
        delete custom_allocator;
    }

    void optimize_for_memory_constrained_environment() {
        // メモリ制約環境向けの設定
        net.opt.workspace_size_mb = 64;   // ワークスペースを削減
        net.opt.use_memory_pool = true;   // メモリプールを有効
        net.opt.use_packing_layout = false;  // パッキングレイアウトを無効（メモリ優先）

        std::cout << "Configured for memory-constrained environment" << std::endl;
    }
};
```

### メモリプール管理

```cpp
class AdvancedMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::time_point<std::chrono::steady_clock> last_used;
    };

    std::vector<MemoryBlock> memory_blocks;
    std::mutex pool_mutex;
    size_t total_pool_size;
    size_t max_pool_size;

public:
    AdvancedMemoryPool(size_t max_size = 512 * 1024 * 1024)  // 512MB default
        : total_pool_size(0), max_pool_size(max_size) {}

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        // 既存ブロックから適切なサイズのものを探す
        for (auto& block : memory_blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                block.last_used = std::chrono::steady_clock::now();
                return block.ptr;
            }
        }

        // 新しいブロックを作成
        if (total_pool_size + size <= max_pool_size) {
            void* ptr = std::aligned_alloc(64, size);  // 64バイトアライメント
            if (ptr) {
                memory_blocks.push_back({ptr, size, true, std::chrono::steady_clock::now()});
                total_pool_size += size;
                return ptr;
            }
        }

        // プールサイズを超える場合は、古いブロックを解放
        cleanup_old_blocks();

        // 再試行
        void* ptr = std::aligned_alloc(64, size);
        if (ptr) {
            memory_blocks.push_back({ptr, size, true, std::chrono::steady_clock::now()});
            total_pool_size += size;
        }

        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        for (auto& block : memory_blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                break;
            }
        }
    }

private:
    void cleanup_old_blocks() {
        auto now = std::chrono::steady_clock::now();
        auto threshold = std::chrono::minutes(5);  // 5分間未使用のブロックを解放

        auto it = memory_blocks.begin();
        while (it != memory_blocks.end()) {
            if (!it->in_use && (now - it->last_used) > threshold) {
                std::free(it->ptr);
                total_pool_size -= it->size;
                it = memory_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

## 8.4 プロファイリングとベンチマーク

性能最適化には正確な測定が不可欠です。包括的なプロファイリングシステムを実装します。

### 詳細なプロファイラー

```cpp
#include <chrono>
#include <map>
#include <iomanip>

class DetailedProfiler {
private:
    struct ProfileData {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::duration<double, std::milli> total_time{0};
        int call_count = 0;
        double min_time = std::numeric_limits<double>::max();
        double max_time = 0.0;
    };

    std::map<std::string, ProfileData> profiles;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> active_timers;
    std::mutex profile_mutex;

public:
    void start_timer(const std::string& name) {
        std::lock_guard<std::mutex> lock(profile_mutex);
        active_timers[name] = std::chrono::high_resolution_clock::now();
    }

    void end_timer(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(profile_mutex);

        auto it = active_timers.find(name);
        if (it != active_timers.end()) {
            auto duration = std::chrono::duration<double, std::milli>(end_time - it->second);
            double ms = duration.count();

            ProfileData& data = profiles[name];
            data.name = name;
            data.total_time += duration;
            data.call_count++;
            data.min_time = std::min(data.min_time, ms);
            data.max_time = std::max(data.max_time, ms);

            active_timers.erase(it);
        }
    }

    void print_results() {
        std::lock_guard<std::mutex> lock(profile_mutex);

        std::cout << "\n=== Performance Profile Results ===" << std::endl;
        std::cout << std::setw(20) << "Operation"
                  << std::setw(10) << "Calls"
                  << std::setw(12) << "Total (ms)"
                  << std::setw(12) << "Avg (ms)"
                  << std::setw(12) << "Min (ms)"
                  << std::setw(12) << "Max (ms)" << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        for (const auto& pair : profiles) {
            const ProfileData& data = pair.second;
            double avg = data.total_time.count() / data.call_count;

            std::cout << std::setw(20) << data.name
                      << std::setw(10) << data.call_count
                      << std::setw(12) << std::fixed << std::setprecision(2) << data.total_time.count()
                      << std::setw(12) << std::fixed << std::setprecision(2) << avg
                      << std::setw(12) << std::fixed << std::setprecision(2) << data.min_time
                      << std::setw(12) << std::fixed << std::setprecision(2) << data.max_time << std::endl;
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(profile_mutex);
        profiles.clear();
        active_timers.clear();
    }
};

// RAII方式のタイマー
class ScopedTimer {
private:
    DetailedProfiler* profiler;
    std::string name;

public:
    ScopedTimer(DetailedProfiler* prof, const std::string& timer_name)
        : profiler(prof), name(timer_name) {
        profiler->start_timer(name);
    }

    ~ScopedTimer() {
        profiler->end_timer(name);
    }
};

#define PROFILE_SCOPE(profiler, name) ScopedTimer timer(profiler, name)
```

### ベンチマークスイート

```cpp
class NCNNBenchmark {
private:
    DetailedProfiler profiler;
    ncnn::Net net;

public:
    struct BenchmarkConfig {
        int warmup_iterations = 10;
        int benchmark_iterations = 100;
        bool measure_memory = true;
        bool measure_cpu_usage = true;
        std::vector<std::pair<int, int>> input_sizes = {{224, 224}, {416, 416}, {640, 640}};
    };

    void run_comprehensive_benchmark(const std::string& model_param,
                                   const std::string& model_bin,
                                   const BenchmarkConfig& config = BenchmarkConfig()) {
        std::cout << "Loading model for benchmark..." << std::endl;

        if (net.load_param(model_param.c_str()) != 0 ||
            net.load_model(model_param.c_str(), model_bin.c_str()) != 0) {
            std::cerr << "Failed to load model" << std::endl;
            return;
        }

        for (const auto& size : config.input_sizes) {
            std::cout << "\nBenchmarking with input size: " << size.first << "x" << size.second << std::endl;
            benchmark_input_size(size.first, size.second, config);
        }

        profiler.print_results();
    }

private:
    void benchmark_input_size(int width, int height, const BenchmarkConfig& config) {
        // ダミー入力データの作成
        ncnn::Mat input(width, height, 3);
        input.fill(0.5f);

        // ウォームアップ
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < config.warmup_iterations; i++) {
            run_single_inference(input, "warmup");
        }

        // ベンチマーク実行
        std::cout << "Running benchmark..." << std::endl;
        std::string benchmark_name = "inference_" + std::to_string(width) + "x" + std::to_string(height);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < config.benchmark_iterations; i++) {
            PROFILE_SCOPE(&profiler, benchmark_name);
            run_single_inference(input, benchmark_name);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double, std::milli>(end_time - start_time);

        std::cout << "Average inference time: "
                  << total_time.count() / config.benchmark_iterations << " ms" << std::endl;
        std::cout << "Throughput: "
                  << (config.benchmark_iterations * 1000.0) / total_time.count() << " FPS" << std::endl;
    }

    void run_single_inference(const ncnn::Mat& input, const std::string& context) {
        PROFILE_SCOPE(&profiler, "preprocess_" + context);

        ncnn::Extractor ex = net.create_extractor();

        {
            PROFILE_SCOPE(&profiler, "input_" + context);
            ex.input("input", input);
        }

        {
            PROFILE_SCOPE(&profiler, "extract_" + context);
            ncnn::Mat output;
            ex.extract("output", output);
        }
    }
};

// 使用例
int main() {
    NCNNBenchmark benchmark;

    NCNNBenchmark::BenchmarkConfig config;
    config.warmup_iterations = 20;
    config.benchmark_iterations = 200;
    config.input_sizes = {{224, 224}, {320, 320}, {416, 416}};

    benchmark.run_comprehensive_benchmark("model.param", "model.bin", config);

    return 0;
}
```

### システムリソース監視

```cpp
#include <fstream>
#include <thread>

class SystemResourceMonitor {
private:
    std::atomic<bool> monitoring;
    std::thread monitor_thread;
    std::vector<double> cpu_usage_history;
    std::vector<size_t> memory_usage_history;

public:
    SystemResourceMonitor() : monitoring(false) {}

    void start_monitoring() {
        monitoring = true;
        monitor_thread = std::thread(&SystemResourceMonitor::monitor_loop, this);
    }

    void stop_monitoring() {
        monitoring = false;
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }

    void print_statistics() {
        if (cpu_usage_history.empty()) {
            std::cout << "No monitoring data available" << std::endl;
            return;
        }

        double avg_cpu = std::accumulate(cpu_usage_history.begin(), cpu_usage_history.end(), 0.0) / cpu_usage_history.size();
        double max_cpu = *std::max_element(cpu_usage_history.begin(), cpu_usage_history.end());

        size_t avg_memory = std::accumulate(memory_usage_history.begin(), memory_usage_history.end(), 0ULL) / memory_usage_history.size();
        size_t max_memory = *std::max_element(memory_usage_history.begin(), memory_usage_history.end());

        std::cout << "\n=== System Resource Usage ===" << std::endl;
        std::cout << "CPU Usage - Average: " << std::fixed << std::setprecision(1) << avg_cpu << "%, Peak: " << max_cpu << "%" << std::endl;
        std::cout << "Memory Usage - Average: " << avg_memory / (1024*1024) << " MB, Peak: " << max_memory / (1024*1024) << " MB" << std::endl;
    }

private:
    void monitor_loop() {
        while (monitoring) {
            double cpu = get_cpu_usage();
            size_t memory = get_memory_usage();

            cpu_usage_history.push_back(cpu);
            memory_usage_history.push_back(memory);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    double get_cpu_usage() {
        // 簡略化されたCPU使用率取得（Linuxの場合）
        static long long prev_idle = 0, prev_total = 0;

        std::ifstream file("/proc/stat");
        std::string line;
        std::getline(file, line);

        std::stringstream ss(line);
        std::string cpu_label;
        long long user, nice, system, idle, iowait, irq, softirq, steal;

        ss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

        long long current_idle = idle + iowait;
        long long current_total = user + nice + system + idle + iowait + irq + softirq + steal;

        long long total_diff = current_total - prev_total;
        long long idle_diff = current_idle - prev_idle;

        double cpu_percent = 0.0;
        if (total_diff > 0) {
            cpu_percent = 100.0 * (total_diff - idle_diff) / total_diff;
        }

        prev_idle = current_idle;
        prev_total = current_total;

        return cpu_percent;
    }

    size_t get_memory_usage() {
        // プロセスのメモリ使用量を取得
        std::ifstream file("/proc/self/status");
        std::string line;

        while (std::getline(file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::stringstream ss(line.substr(6));
                size_t memory_kb;
                ss >> memory_kb;
                return memory_kb * 1024;  // バイト単位で返す
            }
        }

        return 0;
    }
};
```

これらの最適化技術を適用することで、ncnnアプリケーションの性能を大幅に向上させることができます。次章では、これらの技術を統合した実践的なアプリケーション開発について学習します。