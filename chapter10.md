# 10. 高度なトピック

本章では、ncnnの上級者向け機能について詳しく解説します。カスタムレイヤーの実装、動的形状入力への対応、複数モデルの並列実行、ARM NEON最適化など、より高度な技術を習得することで、ncnnの潜在能力を最大限に活用できるようになります。

## 10.1 カスタムレイヤーの実装

ncnnで提供されていない独自の演算を実装したい場合、カスタムレイヤーを作成することができます。

### カスタムレイヤーの基本構造

```cpp
#include <ncnn/layer.h>

class CustomActivationLayer : public ncnn::Layer {
public:
    CustomActivationLayer() {
        one_blob_only = true;  // 入力と出力が1つずつ
        support_inplace = true; // in-place演算をサポート
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        // パラメータの読み込み
        alpha = pd.get(0, 1.0f);  // パラメータ0番目、デフォルト値1.0f
        beta = pd.get(1, 0.0f);   // パラメータ1番目、デフォルト値0.0f
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++) {
            float* ptr = bottom_top_blob.channel(q);

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    // カスタム活性化関数: f(x) = alpha * tanh(beta * x)
                    ptr[x] = alpha * tanh(beta * ptr[x]);
                }
                ptr += w;
            }
        }

        return 0;
    }

private:
    float alpha;
    float beta;
};

DEFINE_LAYER_CREATOR(CustomActivationLayer)
```

### より複雑なカスタムレイヤー（重み付きレイヤー）

```cpp
class CustomConvolutionLayer : public ncnn::Layer {
public:
    CustomConvolutionLayer() {
        one_blob_only = true;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        stride_w = pd.get(2, 1);
        stride_h = pd.get(12, stride_w);
        pad_w = pd.get(3, 0);
        pad_h = pd.get(13, pad_w);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);

        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb) {
        // 重みデータの読み込み
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty()) {
            return -100;
        }

        if (bias_term) {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty()) {
                return -100;
            }
        }

        return 0;
    }

    virtual int create_pipeline(const ncnn::Option& opt) {
        // GPU実装の準備（Vulkan使用時）
        #if NCNN_VULKAN
        if (opt.use_vulkan_compute) {
            return create_pipeline_vulkan(opt);
        }
        #endif

        // CPU実装用の事前計算
        prepare_cpu_implementation();

        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = (w + 2 * pad_w - kernel_w) / stride_w + 1;
        int outh = (h + 2 * pad_h - kernel_h) / stride_h + 1;

        top_blob.create(outw, outh, num_output, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            return -100;
        }

        // カスタム畳み込み演算の実装
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++) {
            float* outptr = top_blob.channel(p);
            const float* weight_ptr = (const float*)weight_data + p * channels * kernel_w * kernel_h;

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    float sum = 0.0f;

                    // 畳み込み演算
                    for (int c = 0; c < channels; c++) {
                        const float* input_ptr = bottom_blob.channel(c);

                        for (int ky = 0; ky < kernel_h; ky++) {
                            for (int kx = 0; kx < kernel_w; kx++) {
                                int input_y = i * stride_h + ky - pad_h;
                                int input_x = j * stride_w + kx - pad_w;

                                if (input_y >= 0 && input_y < h && input_x >= 0 && input_x < w) {
                                    float input_val = input_ptr[input_y * w + input_x];
                                    float weight_val = weight_ptr[c * kernel_h * kernel_w + ky * kernel_w + kx];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }

                    // バイアス項の追加
                    if (bias_term) {
                        const float* bias_ptr = (const float*)bias_data;
                        sum += bias_ptr[p];
                    }

                    outptr[j] = sum;
                }
                outptr += outw;
            }
        }

        return 0;
    }

private:
    // レイヤーパラメータ
    int num_output;
    int kernel_w, kernel_h;
    int stride_w, stride_h;
    int pad_w, pad_h;
    int bias_term;
    int weight_data_size;

    // モデルデータ
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;

    void prepare_cpu_implementation() {
        // CPU用の最適化準備（例：重みの再配置）
    }

    #if NCNN_VULKAN
    int create_pipeline_vulkan(const ncnn::Option& opt) {
        // Vulkan GPU実装の準備
        return 0;
    }
    #endif
};

DEFINE_LAYER_CREATOR(CustomConvolutionLayer)
```

### カスタムレイヤーの登録と使用

```cpp
class CustomLayerRegistry {
public:
    static void register_custom_layers() {
        // カスタムレイヤーの登録
        ncnn::layer_registry[ncnn::LayerType::CustomActivation] = CustomActivationLayer_layer_creator;
        ncnn::layer_registry[ncnn::LayerType::CustomConvolution] = CustomConvolutionLayer_layer_creator;
    }

    static ncnn::Layer* create_custom_layer(const std::string& type_name) {
        if (type_name == "CustomActivation") {
            return new CustomActivationLayer();
        } else if (type_name == "CustomConvolution") {
            return new CustomConvolutionLayer();
        }
        return nullptr;
    }
};

class NetworkWithCustomLayers {
private:
    ncnn::Net net;

public:
    bool load_model_with_custom_layers(const std::string& param_path, const std::string& bin_path) {
        // カスタムレイヤーの登録
        CustomLayerRegistry::register_custom_layers();

        // カスタムレイヤーファクトリーの設定
        net.register_custom_layer("CustomActivation", CustomActivationLayer_layer_creator);
        net.register_custom_layer("CustomConvolution", CustomConvolutionLayer_layer_creator);

        // モデル読み込み
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        return (ret1 == 0 && ret2 == 0);
    }

    ncnn::Mat inference(const ncnn::Mat& input) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", input);

        ncnn::Mat output;
        ex.extract("output", output);

        return output;
    }
};
```

## 10.2 動的形状入力への対応

入力サイズが実行時に決まる場合の対応方法を実装します。

```cpp
class DynamicShapeInference {
private:
    ncnn::Net net;
    std::map<std::string, std::vector<int>> input_shape_cache;
    std::mutex shape_cache_mutex;

public:
    DynamicShapeInference() {
        // 動的形状対応の設定
        net.opt.use_packing_layout = false;  // パッキングレイアウトを無効
        net.opt.use_memory_pool = false;     // メモリプールを無効（動的サイズのため）
    }

    bool load_model(const std::string& param_path, const std::string& bin_path) {
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        return (ret1 == 0 && ret2 == 0);
    }

    ncnn::Mat inference_dynamic(const ncnn::Mat& input, const std::string& input_name = "input") {
        // 入力形状のキャッシュキー作成
        std::string cache_key = create_shape_key(input);

        // 同じ形状での実行履歴確認
        {
            std::lock_guard<std::mutex> lock(shape_cache_mutex);
            if (input_shape_cache.find(cache_key) == input_shape_cache.end()) {
                input_shape_cache[cache_key] = {input.w, input.h, input.c};
                std::cout << "New input shape registered: " << cache_key << std::endl;
            }
        }

        // 推論実行
        ncnn::Extractor ex = net.create_extractor();

        // 動的入力の設定
        ex.input(input_name.c_str(), input);

        ncnn::Mat output;
        ex.extract("output", output);

        return output;
    }

    // 複数の異なるサイズの入力に対応
    std::vector<ncnn::Mat> batch_inference_dynamic(const std::vector<ncnn::Mat>& inputs) {
        std::vector<ncnn::Mat> outputs;

        for (const auto& input : inputs) {
            try {
                ncnn::Mat output = inference_dynamic(input);
                outputs.push_back(output);
            } catch (const std::exception& e) {
                std::cerr << "Dynamic inference failed for input shape ["
                         << input.c << "," << input.h << "," << input.w << "]: "
                         << e.what() << std::endl;
                // 失敗した場合は空のMatを追加
                outputs.push_back(ncnn::Mat());
            }
        }

        return outputs;
    }

    // アスペクト比を保持した動的リサイズ
    ncnn::Mat adaptive_resize_and_infer(const cv::Mat& image, int target_size = 640) {
        // アスペクト比を保持したリサイズ
        int orig_w = image.cols;
        int orig_h = image.rows;

        float scale = std::min((float)target_size / orig_w, (float)target_size / orig_h);
        int new_w = (int)(orig_w * scale);
        int new_h = (int)(orig_h * scale);

        // 32の倍数に調整（一般的なCNNの制約）
        new_w = (new_w + 31) / 32 * 32;
        new_h = (new_h + 31) / 32 * 32;

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));

        // ncnn::Matに変換
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        ncnn::Mat input = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, new_w, new_h);

        // 正規化
        const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        input.substract_mean_normalize(0, norm_vals);

        return inference_dynamic(input);
    }

    void print_shape_statistics() {
        std::lock_guard<std::mutex> lock(shape_cache_mutex);

        std::cout << "\n=== Dynamic Shape Statistics ===" << std::endl;
        std::cout << "Unique input shapes encountered: " << input_shape_cache.size() << std::endl;

        for (const auto& pair : input_shape_cache) {
            const auto& shape = pair.second;
            std::cout << "Shape: [" << shape[2] << "," << shape[1] << "," << shape[0] << "]" << std::endl;
        }
    }

private:
    std::string create_shape_key(const ncnn::Mat& input) {
        return std::to_string(input.c) + "x" + std::to_string(input.h) + "x" + std::to_string(input.w);
    }
};
```

## 10.3 複数モデルの並列実行

複数のモデルを同時に実行して、より高度な推論パイプラインを構築します。

```cpp
#include <future>
#include <queue>

class MultiModelInference {
private:
    struct ModelInstance {
        std::unique_ptr<ncnn::Net> net;
        std::string model_name;
        std::mutex net_mutex;
        std::atomic<int> active_inferences;

        ModelInstance(const std::string& name) : model_name(name), active_inferences(0) {
            net = std::make_unique<ncnn::Net>();
        }
    };

    std::vector<std::unique_ptr<ModelInstance>> models;
    std::vector<std::thread> worker_threads;
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> running;

public:
    MultiModelInference(int num_workers = 4) : running(false) {
        start_worker_threads(num_workers);
    }

    ~MultiModelInference() {
        stop_worker_threads();
    }

    bool add_model(const std::string& model_name, const std::string& param_path, const std::string& bin_path) {
        auto model_instance = std::make_unique<ModelInstance>(model_name);

        // 各モデルで独立した設定
        model_instance->net->opt.use_vulkan_compute = false;  // マルチモデルではCPUを推奨
        model_instance->net->opt.num_threads = 1;  // ワーカースレッド毎に1スレッド

        int ret1 = model_instance->net->load_param(param_path.c_str());
        int ret2 = model_instance->net->load_model(param_path.c_str(), bin_path.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load model: " << model_name << std::endl;
            return false;
        }

        models.push_back(std::move(model_instance));
        std::cout << "Model added: " << model_name << std::endl;
        return true;
    }

    // 非同期推論
    std::future<ncnn::Mat> inference_async(const std::string& model_name, const ncnn::Mat& input,
                                         const std::string& input_layer = "input",
                                         const std::string& output_layer = "output") {
        auto promise = std::make_shared<std::promise<ncnn::Mat>>();
        auto future = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push([this, model_name, input, input_layer, output_layer, promise]() {
                try {
                    ncnn::Mat result = execute_inference(model_name, input, input_layer, output_layer);
                    promise->set_value(result);
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }

        queue_cv.notify_one();
        return future;
    }

    // 並列推論（複数モデルで同じ入力を処理）
    std::map<std::string, ncnn::Mat> parallel_inference(const ncnn::Mat& input,
                                                       const std::vector<std::string>& model_names) {
        std::vector<std::future<std::pair<std::string, ncnn::Mat>>> futures;

        // 各モデルで非同期推論を開始
        for (const auto& model_name : model_names) {
            auto future = std::async(std::launch::async, [this, model_name, input]() {
                ncnn::Mat result = execute_inference(model_name, input);
                return std::make_pair(model_name, result);
            });
            futures.push_back(std::move(future));
        }

        // 結果の収集
        std::map<std::string, ncnn::Mat> results;
        for (auto& future : futures) {
            try {
                auto pair = future.get();
                results[pair.first] = pair.second;
            } catch (const std::exception& e) {
                std::cerr << "Parallel inference error: " << e.what() << std::endl;
            }
        }

        return results;
    }

    // パイプライン推論（モデル間でデータを順次処理）
    ncnn::Mat pipeline_inference(const ncnn::Mat& input, const std::vector<std::string>& model_pipeline) {
        ncnn::Mat current_input = input;

        for (const auto& model_name : model_pipeline) {
            current_input = execute_inference(model_name, current_input);
        }

        return current_input;
    }

    // アンサンブル推論（複数モデルの結果を統合）
    ncnn::Mat ensemble_inference(const ncnn::Mat& input, const std::vector<std::string>& model_names,
                               const std::string& ensemble_method = "average") {
        auto results = parallel_inference(input, model_names);

        if (results.empty()) {
            throw std::runtime_error("No inference results for ensemble");
        }

        // 最初の結果の形状を参考にする
        auto first_result = results.begin()->second;
        ncnn::Mat ensemble_result(first_result.w, first_result.h, first_result.c);
        ensemble_result.fill(0.0f);

        if (ensemble_method == "average") {
            // 平均化
            for (const auto& pair : results) {
                const ncnn::Mat& result = pair.second;
                const float* src_data = (const float*)result.data;
                float* dst_data = (float*)ensemble_result.data;

                size_t total_elements = result.total();
                for (size_t i = 0; i < total_elements; i++) {
                    dst_data[i] += src_data[i];
                }
            }

            // 平均値計算
            float* dst_data = (float*)ensemble_result.data;
            size_t total_elements = ensemble_result.total();
            float num_models = static_cast<float>(results.size());

            for (size_t i = 0; i < total_elements; i++) {
                dst_data[i] /= num_models;
            }

        } else if (ensemble_method == "max") {
            // 最大値選択
            bool first = true;
            for (const auto& pair : results) {
                const ncnn::Mat& result = pair.second;
                const float* src_data = (const float*)result.data;
                float* dst_data = (float*)ensemble_result.data;

                size_t total_elements = result.total();
                for (size_t i = 0; i < total_elements; i++) {
                    if (first || src_data[i] > dst_data[i]) {
                        dst_data[i] = src_data[i];
                    }
                }
                first = false;
            }
        }

        return ensemble_result;
    }

    void print_model_statistics() {
        std::cout << "\n=== Multi-Model Statistics ===" << std::endl;
        for (const auto& model : models) {
            std::cout << "Model: " << model->model_name
                     << ", Active inferences: " << model->active_inferences.load() << std::endl;
        }
    }

private:
    ncnn::Mat execute_inference(const std::string& model_name, const ncnn::Mat& input,
                              const std::string& input_layer = "input",
                              const std::string& output_layer = "output") {
        // モデルの検索
        auto model_it = std::find_if(models.begin(), models.end(),
                                   [&model_name](const std::unique_ptr<ModelInstance>& model) {
                                       return model->model_name == model_name;
                                   });

        if (model_it == models.end()) {
            throw std::runtime_error("Model not found: " + model_name);
        }

        ModelInstance* model = model_it->get();
        model->active_inferences.fetch_add(1);

        try {
            std::lock_guard<std::mutex> lock(model->net_mutex);

            ncnn::Extractor ex = model->net->create_extractor();
            ex.input(input_layer.c_str(), input);

            ncnn::Mat output;
            ex.extract(output_layer.c_str(), output);

            model->active_inferences.fetch_sub(1);
            return output;

        } catch (...) {
            model->active_inferences.fetch_sub(1);
            throw;
        }
    }

    void start_worker_threads(int num_workers) {
        running = true;

        for (int i = 0; i < num_workers; i++) {
            worker_threads.emplace_back([this]() {
                while (running) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        queue_cv.wait(lock, [this] { return !task_queue.empty() || !running; });

                        if (!running) break;

                        task = task_queue.front();
                        task_queue.pop();
                    }

                    task();
                }
            });
        }
    }

    void stop_worker_threads() {
        running = false;
        queue_cv.notify_all();

        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
};
```

## 10.4 ARM NEON最適化

ARM プロセッサでの性能を最大化するためのNEON最適化技術を実装します。

```cpp
#ifdef __ARM_NEON
#include <arm_neon.h>

class NEONOptimizedOperations {
public:
    // NEON最適化された要素ごとの加算
    static void add_arrays_neon(const float* a, const float* b, float* result, int size) {
        int neon_size = size - (size % 4);  // 4要素ずつ処理

        // NEON演算（4要素並列）
        for (int i = 0; i < neon_size; i += 4) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vresult = vaddq_f32(va, vb);
            vst1q_f32(&result[i], vresult);
        }

        // 残りの要素をスカラー処理
        for (int i = neon_size; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }

    // NEON最適化されたReLU活性化関数
    static void relu_neon(const float* input, float* output, int size) {
        int neon_size = size - (size % 4);
        float32x4_t zero = vdupq_n_f32(0.0f);

        for (int i = 0; i < neon_size; i += 4) {
            float32x4_t vin = vld1q_f32(&input[i]);
            float32x4_t vout = vmaxq_f32(vin, zero);
            vst1q_f32(&output[i], vout);
        }

        for (int i = neon_size; i < size; i++) {
            output[i] = std::max(0.0f, input[i]);
        }
    }

    // NEON最適化された畳み込み演算（簡略版）
    static void conv2d_3x3_neon(const float* input, const float* kernel, float* output,
                               int input_w, int input_h, int output_w, int output_h) {
        for (int y = 0; y < output_h; y++) {
            for (int x = 0; x < output_w; x += 4) {  // 4ピクセルずつ処理
                int remaining = std::min(4, output_w - x);

                if (remaining == 4) {
                    // 4ピクセル並列処理
                    float32x4_t sum = vdupq_n_f32(0.0f);

                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int input_y = y + ky;
                            int input_x_base = x + kx;

                            if (input_x_base + 3 < input_w && input_y < input_h) {
                                float32x4_t input_vals = vld1q_f32(&input[input_y * input_w + input_x_base]);
                                float kernel_val = kernel[ky * 3 + kx];
                                float32x4_t kernel_vec = vdupq_n_f32(kernel_val);
                                sum = vmlaq_f32(sum, input_vals, kernel_vec);
                            }
                        }
                    }

                    vst1q_f32(&output[y * output_w + x], sum);
                } else {
                    // 残りはスカラー処理
                    for (int i = 0; i < remaining; i++) {
                        float sum = 0.0f;
                        for (int ky = 0; ky < 3; ky++) {
                            for (int kx = 0; kx < 3; kx++) {
                                int input_y = y + ky;
                                int input_x = x + i + kx;
                                if (input_x < input_w && input_y < input_h) {
                                    sum += input[input_y * input_w + input_x] * kernel[ky * 3 + kx];
                                }
                            }
                        }
                        output[y * output_w + x + i] = sum;
                    }
                }
            }
        }
    }

    // NEON最適化されたSoftmax
    static void softmax_neon(const float* input, float* output, int size) {
        // 最大値を見つける（数値安定性のため）
        float max_val = *std::max_element(input, input + size);

        // exp計算と合計
        float sum = 0.0f;
        int neon_size = size - (size % 4);
        float32x4_t max_vec = vdupq_n_f32(max_val);
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        for (int i = 0; i < neon_size; i += 4) {
            float32x4_t input_vec = vld1q_f32(&input[i]);
            float32x4_t shifted = vsubq_f32(input_vec, max_vec);

            // exp近似（簡略版、実際にはより精密な実装が必要）
            float exp_vals[4];
            vst1q_f32(exp_vals, shifted);
            for (int j = 0; j < 4; j++) {
                exp_vals[j] = std::exp(exp_vals[j]);
            }

            float32x4_t exp_vec = vld1q_f32(exp_vals);
            vst1q_f32(&output[i], exp_vec);
            sum_vec = vaddq_f32(sum_vec, exp_vec);
        }

        // sum_vecの合計
        float sum_array[4];
        vst1q_f32(sum_array, sum_vec);
        sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // 残りの要素
        for (int i = neon_size; i < size; i++) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }

        // 正規化
        float32x4_t sum_vec_inv = vdupq_n_f32(1.0f / sum);
        for (int i = 0; i < neon_size; i += 4) {
            float32x4_t output_vec = vld1q_f32(&output[i]);
            output_vec = vmulq_f32(output_vec, sum_vec_inv);
            vst1q_f32(&output[i], output_vec);
        }

        for (int i = neon_size; i < size; i++) {
            output[i] /= sum;
        }
    }
};

class NEONOptimizedCustomLayer : public ncnn::Layer {
public:
    NEONOptimizedCustomLayer() {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const {
        int size = bottom_top_blob.total();
        float* data = (float*)bottom_top_blob.data;

        // NEON最適化されたReLUを適用
        NEONOptimizedOperations::relu_neon(data, data, size);

        return 0;
    }
};

// NEON最適化のベンチマーク
class NEONBenchmark {
public:
    static void benchmark_operations() {
        const int size = 1000000;
        std::vector<float> a(size), b(size), result_scalar(size), result_neon(size);

        // ランダムデータの生成
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < size; i++) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }

        // スカラー実装のベンチマーク
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++) {
            result_scalar[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start);

        // NEON実装のベンチマーク
        start = std::chrono::high_resolution_clock::now();
        NEONOptimizedOperations::add_arrays_neon(a.data(), b.data(), result_neon.data(), size);
        end = std::chrono::high_resolution_clock::now();
        auto neon_time = std::chrono::duration<double, std::milli>(end - start);

        std::cout << "=== NEON Optimization Benchmark ===" << std::endl;
        std::cout << "Array size: " << size << " elements" << std::endl;
        std::cout << "Scalar time: " << scalar_time.count() << " ms" << std::endl;
        std::cout << "NEON time: " << neon_time.count() << " ms" << std::endl;
        std::cout << "Speedup: " << scalar_time.count() / neon_time.count() << "x" << std::endl;

        // 結果の検証
        float max_diff = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = std::abs(result_scalar[i] - result_neon[i]);
            max_diff = std::max(max_diff, diff);
        }
        std::cout << "Maximum difference: " << max_diff << std::endl;
    }
};

#endif // __ARM_NEON
```

これらの高度な技術を習得することで、ncnnの潜在能力を最大限に引き出し、特定の要件に合わせた最適化されたアプリケーションを開発することができます。次章では、開発時によく遭遇する問題とその解決方法について学習します。