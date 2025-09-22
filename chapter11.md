# 11. トラブルシューティング

ncnn開発において頻繁に遭遇する問題とその解決方法を体系的にまとめます。本章では、よくある問題、デバッグテクニック、パフォーマンス問題の診断について詳しく解説します。

## 11.1 よくある問題と解決方法

### モデル読み込み関連の問題

**問題1: モデルファイルの読み込みに失敗する**

```cpp
// 症状：net.load_param() または net.load_model() が非ゼロを返す

class ModelLoadingTroubleshooter {
public:
    static void diagnose_loading_error(const std::string& param_path, const std::string& bin_path) {
        std::cout << "=== Model Loading Diagnosis ===" << std::endl;

        // ファイル存在確認
        if (!check_file_exists(param_path)) {
            std::cout << "ERROR: Parameter file not found: " << param_path << std::endl;
            return;
        }

        if (!check_file_exists(bin_path)) {
            std::cout << "ERROR: Binary file not found: " << bin_path << std::endl;
            return;
        }

        std::cout << "✓ Both files exist" << std::endl;

        // ファイルサイズ確認
        auto param_size = get_file_size(param_path);
        auto bin_size = get_file_size(bin_path);

        std::cout << "Parameter file size: " << param_size << " bytes" << std::endl;
        std::cout << "Binary file size: " << bin_size << " bytes" << std::endl;

        if (param_size == 0) {
            std::cout << "ERROR: Parameter file is empty" << std::endl;
            return;
        }

        if (bin_size == 0) {
            std::cout << "ERROR: Binary file is empty" << std::endl;
            return;
        }

        // パラメータファイルの構文確認
        if (!validate_param_file(param_path)) {
            std::cout << "ERROR: Invalid parameter file format" << std::endl;
            return;
        }

        std::cout << "✓ Parameter file format is valid" << std::endl;

        // バイナリファイルのマジックナンバー確認
        if (!validate_bin_file(bin_path)) {
            std::cout << "ERROR: Invalid binary file format" << std::endl;
            return;
        }

        std::cout << "✓ Binary file format is valid" << std::endl;

        // 実際の読み込みテスト
        test_actual_loading(param_path, bin_path);
    }

private:
    static bool check_file_exists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }

    static size_t get_file_size(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        return file.tellg();
    }

    static bool validate_param_file(const std::string& path) {
        std::ifstream file(path);
        std::string line;

        // 最初の行：マジックナンバー
        if (!std::getline(file, line)) {
            return false;
        }

        try {
            int magic = std::stoi(line);
            if (magic != 7767517) {
                std::cout << "WARNING: Unexpected magic number: " << magic << std::endl;
            }
        } catch (...) {
            std::cout << "ERROR: Invalid magic number format" << std::endl;
            return false;
        }

        // 2行目：レイヤー数とブロブ数
        if (!std::getline(file, line)) {
            return false;
        }

        std::istringstream iss(line);
        int layer_count, blob_count;
        if (!(iss >> layer_count >> blob_count)) {
            std::cout << "ERROR: Invalid layer/blob count format" << std::endl;
            return false;
        }

        std::cout << "Model info: " << layer_count << " layers, " << blob_count << " blobs" << std::endl;
        return true;
    }

    static bool validate_bin_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }

        // 最初の数バイトを読んで妥当性を確認
        char buffer[16];
        file.read(buffer, sizeof(buffer));

        return file.gcount() > 0;  // 何らかのデータが読めた
    }

    static void test_actual_loading(const std::string& param_path, const std::string& bin_path) {
        ncnn::Net test_net;

        int ret1 = test_net.load_param(param_path.c_str());
        if (ret1 != 0) {
            std::cout << "ERROR: load_param failed with code: " << ret1 << std::endl;
            std::cout << "Common causes: "
                     << "Corrupted parameter file, unsupported layer type, or memory allocation failure" << std::endl;
            return;
        }

        int ret2 = test_net.load_model(param_path.c_str(), bin_path.c_str());
        if (ret2 != 0) {
            std::cout << "ERROR: load_model failed with code: " << ret2 << std::endl;
            std::cout << "Common causes: "
                     << "Mismatched parameter and binary files, corrupted binary file, or insufficient memory" << std::endl;
            return;
        }

        std::cout << "✓ Model loading successful" << std::endl;
    }
};
```

**問題2: 推論時の形状エラー**

```cpp
class ShapeErrorDiagnostic {
public:
    static void diagnose_shape_mismatch(const ncnn::Net& net, const ncnn::Mat& input,
                                      const std::string& input_name) {
        std::cout << "=== Shape Mismatch Diagnosis ===" << std::endl;

        // 入力テンソルの情報表示
        std::cout << "Input tensor info:" << std::endl;
        std::cout << "  Dimensions: " << input.dims << std::endl;
        std::cout << "  Shape: [";
        if (input.dims >= 1) std::cout << input.w;
        if (input.dims >= 2) std::cout << ", " << input.h;
        if (input.dims >= 3) std::cout << ", " << input.c;
        std::cout << "]" << std::endl;
        std::cout << "  Element size: " << input.elemsize << " bytes" << std::endl;
        std::cout << "  Total elements: " << input.total() << std::endl;

        // データの妥当性確認
        validate_input_data(input);

        // 推論テスト
        test_inference_with_debug(net, input, input_name);
    }

private:
    static void validate_input_data(const ncnn::Mat& input) {
        if (input.empty()) {
            std::cout << "ERROR: Input tensor is empty" << std::endl;
            return;
        }

        const float* data = (const float*)input.data;
        size_t total = input.total();

        int nan_count = 0;
        int inf_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < total; i++) {
            float val = data[i];

            if (std::isnan(val)) {
                nan_count++;
            } else if (std::isinf(val)) {
                inf_count++;
            } else {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }

        std::cout << "Input data validation:" << std::endl;
        std::cout << "  NaN values: " << nan_count << std::endl;
        std::cout << "  Inf values: " << inf_count << std::endl;
        std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;

        if (nan_count > 0 || inf_count > 0) {
            std::cout << "WARNING: Input contains invalid values" << std::endl;
        }
    }

    static void test_inference_with_debug(const ncnn::Net& net, const ncnn::Mat& input,
                                        const std::string& input_name) {
        try {
            ncnn::Extractor ex = net.create_extractor();

            std::cout << "Attempting to set input '" << input_name << "'..." << std::endl;
            int ret = ex.input(input_name.c_str(), input);

            if (ret != 0) {
                std::cout << "ERROR: Failed to set input (code: " << ret << ")" << std::endl;
                std::cout << "Possible causes:" << std::endl;
                std::cout << "  - Input layer name '" << input_name << "' not found" << std::endl;
                std::cout << "  - Input shape mismatch" << std::endl;
                std::cout << "  - Memory allocation failure" << std::endl;
                return;
            }

            std::cout << "✓ Input set successfully" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "EXCEPTION during input setting: " << e.what() << std::endl;
        }
    }
};
```

### メモリ関連の問題

**問題3: メモリ不足エラー**

```cpp
class MemoryDiagnostic {
public:
    static void diagnose_memory_issues() {
        std::cout << "=== Memory Diagnostic ===" << std::endl;

        // システムメモリ情報
        print_system_memory_info();

        // プロセスメモリ使用量
        print_process_memory_info();

        // ncnn設定の確認
        print_ncnn_memory_config();

        // メモリリーク検出
        detect_memory_leaks();
    }

private:
    static void print_system_memory_info() {
        std::cout << "System Memory Info:" << std::endl;

#ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;

        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0 || line.find("MemAvailable:") == 0 ||
                line.find("MemFree:") == 0) {
                std::cout << "  " << line << std::endl;
            }
        }
#endif

        // 汎用的なメモリ情報取得
        try {
            std::cout << "  Hardware concurrency: " << std::thread::hardware_concurrency() << std::endl;
        } catch (...) {
            std::cout << "  Unable to determine hardware info" << std::endl;
        }
    }

    static void print_process_memory_info() {
        std::cout << "Process Memory Info:" << std::endl;

#ifdef __linux__
        std::ifstream status("/proc/self/status");
        std::string line;

        while (std::getline(status, line)) {
            if (line.find("VmSize:") == 0 || line.find("VmRSS:") == 0 ||
                line.find("VmPeak:") == 0 || line.find("VmHWM:") == 0) {
                std::cout << "  " << line << std::endl;
            }
        }
#endif
    }

    static void print_ncnn_memory_config() {
        std::cout << "NCNN Memory Configuration:" << std::endl;

        // デフォルトのオプションを作成して設定を確認
        ncnn::Option opt;
        std::cout << "  Blob allocator: " << (opt.blob_allocator ? "Custom" : "Default") << std::endl;
        std::cout << "  Workspace allocator: " << (opt.workspace_allocator ? "Custom" : "Default") << std::endl;
        std::cout << "  Use memory pool: " << (opt.use_memory_pool ? "Yes" : "No") << std::endl;

#if NCNN_VULKAN
        std::cout << "  Vulkan compute: " << (opt.use_vulkan_compute ? "Enabled" : "Disabled") << std::endl;
        if (opt.use_vulkan_compute && ncnn::get_gpu_count() > 0) {
            const ncnn::GpuInfo& gpu_info = ncnn::get_gpu_info(0);
            std::cout << "  GPU device: " << gpu_info.device_name() << std::endl;
        }
#endif
    }

    static void detect_memory_leaks() {
        std::cout << "Memory Leak Detection:" << std::endl;

        // 簡単なメモリリーク検出
        size_t initial_memory = get_current_memory_usage();

        {
            // テストスコープ
            ncnn::Net test_net;
            ncnn::Mat test_mat(100, 100, 3);
            test_mat.fill(1.0f);

            // スコープ終了でリソースが解放されるはず
        }

        // ガベージコレクションを促進（可能であれば）
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        size_t final_memory = get_current_memory_usage();
        long memory_diff = final_memory - initial_memory;

        std::cout << "  Initial memory: " << initial_memory / 1024 << " KB" << std::endl;
        std::cout << "  Final memory: " << final_memory / 1024 << " KB" << std::endl;
        std::cout << "  Difference: " << memory_diff / 1024 << " KB" << std::endl;

        if (memory_diff > 1024) {  // 1KB以上の差
            std::cout << "  WARNING: Possible memory leak detected" << std::endl;
        } else {
            std::cout << "  ✓ No significant memory leak detected" << std::endl;
        }
    }

    static size_t get_current_memory_usage() {
#ifdef __linux__
        std::ifstream status("/proc/self/status");
        std::string line;

        while (std::getline(status, line)) {
            if (line.find("VmRSS:") == 0) {
                std::stringstream ss(line);
                std::string label, kb_str;
                size_t kb;
                ss >> label >> kb;
                return kb * 1024;  // バイト単位で返す
            }
        }
#endif
        return 0;
    }
};
```

## 11.2 デバッグテクニック

### 推論プロセスの詳細ログ

```cpp
class InferenceDebugger {
private:
    ncnn::Net& net;
    bool debug_enabled;
    std::ofstream debug_log;

public:
    InferenceDebugger(ncnn::Net& network, bool enable_debug = true)
        : net(network), debug_enabled(enable_debug) {
        if (debug_enabled) {
            debug_log.open("ncnn_debug_" + get_timestamp() + ".log");
        }
    }

    ~InferenceDebugger() {
        if (debug_log.is_open()) {
            debug_log.close();
        }
    }

    ncnn::Mat debug_inference(const ncnn::Mat& input, const std::string& input_name = "input",
                            const std::string& output_name = "output") {
        if (!debug_enabled) {
            // デバッグ無効時は通常の推論
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_name.c_str(), input);

            ncnn::Mat output;
            ex.extract(output_name.c_str(), output);
            return output;
        }

        log("=== Starting Debug Inference ===");
        log("Input name: " + input_name);
        log("Output name: " + output_name);
        log_tensor_info("Input", input);

        ncnn::Extractor ex = net.create_extractor();

        // デバッグ機能を有効にしたエクストラクター
        ex.set_light_mode(false);  // 軽量モードを無効にしてデバッグ情報を保持

        auto start_time = std::chrono::high_resolution_clock::now();

        // 入力設定
        log("Setting input...");
        int input_ret = ex.input(input_name.c_str(), input);
        if (input_ret != 0) {
            log("ERROR: Failed to set input, return code: " + std::to_string(input_ret));
            throw std::runtime_error("Input setting failed");
        }
        log("✓ Input set successfully");

        // 中間層の出力を取得（可能であれば）
        extract_intermediate_outputs(ex);

        // 最終出力の取得
        log("Extracting output...");
        ncnn::Mat output;
        int extract_ret = ex.extract(output_name.c_str(), output);
        if (extract_ret != 0) {
            log("ERROR: Failed to extract output, return code: " + std::to_string(extract_ret));
            throw std::runtime_error("Output extraction failed");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);

        log("✓ Output extracted successfully");
        log_tensor_info("Output", output);
        log("Total inference time: " + std::to_string(duration.count()) + " ms");
        log("=== Debug Inference Complete ===\n");

        return output;
    }

    void set_layer_debug(const std::vector<std::string>& layer_names) {
        debug_layer_names = layer_names;
        log("Debug enabled for layers: " + join_strings(layer_names, ", "));
    }

    void dump_layer_weights(const std::string& layer_name) {
        log("=== Layer Weights Dump: " + layer_name + " ===");
        // 注意: ncnnでは直接的にレイヤーの重みにアクセスする標準APIは限定的
        // この機能は内部構造に依存するため、実装は複雑になります
        log("Weight dumping feature requires advanced ncnn internals access");
    }

private:
    std::vector<std::string> debug_layer_names;

    void log(const std::string& message) {
        if (!debug_enabled) return;

        std::string timestamp = get_timestamp();
        std::string log_line = "[" + timestamp + "] " + message;

        std::cout << log_line << std::endl;
        if (debug_log.is_open()) {
            debug_log << log_line << std::endl;
            debug_log.flush();
        }
    }

    void log_tensor_info(const std::string& name, const ncnn::Mat& tensor) {
        if (tensor.empty()) {
            log(name + " tensor: EMPTY");
            return;
        }

        std::stringstream ss;
        ss << name << " tensor info:";
        ss << " dims=" << tensor.dims;
        ss << " shape=[";
        if (tensor.dims >= 1) ss << tensor.w;
        if (tensor.dims >= 2) ss << "," << tensor.h;
        if (tensor.dims >= 3) ss << "," << tensor.c;
        ss << "]";
        ss << " elemsize=" << tensor.elemsize;
        ss << " total=" << tensor.total();

        log(ss.str());

        // 統計情報
        log_tensor_statistics(name, tensor);
    }

    void log_tensor_statistics(const std::string& name, const ncnn::Mat& tensor) {
        if (tensor.empty()) return;

        const float* data = (const float*)tensor.data;
        size_t total = tensor.total();

        if (total == 0) return;

        float min_val = data[0];
        float max_val = data[0];
        double sum = 0.0;
        int zero_count = 0;
        int nan_count = 0;

        for (size_t i = 0; i < total; i++) {
            float val = data[i];

            if (std::isnan(val)) {
                nan_count++;
                continue;
            }

            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;

            if (val == 0.0f) {
                zero_count++;
            }
        }

        double mean = sum / (total - nan_count);

        // 分散計算
        double variance = 0.0;
        for (size_t i = 0; i < total; i++) {
            if (!std::isnan(data[i])) {
                variance += (data[i] - mean) * (data[i] - mean);
            }
        }
        variance /= (total - nan_count);

        std::stringstream ss;
        ss << name << " statistics:";
        ss << " min=" << min_val;
        ss << " max=" << max_val;
        ss << " mean=" << std::fixed << std::setprecision(6) << mean;
        ss << " std=" << std::sqrt(variance);
        ss << " zeros=" << zero_count;
        ss << " nans=" << nan_count;

        log(ss.str());
    }

    void extract_intermediate_outputs(ncnn::Extractor& ex) {
        for (const auto& layer_name : debug_layer_names) {
            try {
                ncnn::Mat intermediate;
                int ret = ex.extract(layer_name.c_str(), intermediate);

                if (ret == 0) {
                    log_tensor_info("Intermediate[" + layer_name + "]", intermediate);
                } else {
                    log("Failed to extract intermediate output from: " + layer_name);
                }
            } catch (...) {
                log("Exception while extracting: " + layer_name);
            }
        }
    }

    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string join_strings(const std::vector<std::string>& strings, const std::string& delimiter) {
        if (strings.empty()) return "";

        std::string result = strings[0];
        for (size_t i = 1; i < strings.size(); i++) {
            result += delimiter + strings[i];
        }
        return result;
    }
};
```

### モデル構造の可視化

```cpp
class ModelAnalyzer {
public:
    static void analyze_model_structure(const std::string& param_path) {
        std::cout << "=== Model Structure Analysis ===" << std::endl;

        std::ifstream param_file(param_path);
        if (!param_file.is_open()) {
            std::cout << "ERROR: Cannot open parameter file: " << param_path << std::endl;
            return;
        }

        // ヘッダー情報の読み取り
        std::string line;
        std::getline(param_file, line);  // マジックナンバー
        int magic = std::stoi(line);

        std::getline(param_file, line);  // レイヤー数とブロブ数
        std::istringstream header_iss(line);
        int layer_count, blob_count;
        header_iss >> layer_count >> blob_count;

        std::cout << "Magic number: " << magic << std::endl;
        std::cout << "Layer count: " << layer_count << std::endl;
        std::cout << "Blob count: " << blob_count << std::endl;
        std::cout << std::endl;

        // レイヤー情報の解析
        std::map<std::string, int> layer_type_count;
        std::vector<LayerInfo> layers;

        for (int i = 0; i < layer_count; i++) {
            if (!std::getline(param_file, line)) {
                break;
            }

            LayerInfo layer_info = parse_layer_line(line);
            layers.push_back(layer_info);

            layer_type_count[layer_info.type]++;
        }

        // 統計情報の表示
        print_layer_statistics(layer_type_count);
        print_detailed_layer_info(layers);
        analyze_model_complexity(layers);
    }

private:
    struct LayerInfo {
        std::string type;
        std::string name;
        int input_count;
        int output_count;
        std::vector<std::string> input_blobs;
        std::vector<std::string> output_blobs;
        std::map<int, std::string> parameters;
    };

    static LayerInfo parse_layer_line(const std::string& line) {
        LayerInfo info;
        std::istringstream iss(line);

        iss >> info.type >> info.name >> info.input_count >> info.output_count;

        // 入力ブロブ名
        for (int i = 0; i < info.input_count; i++) {
            std::string blob_name;
            iss >> blob_name;
            info.input_blobs.push_back(blob_name);
        }

        // 出力ブロブ名
        for (int i = 0; i < info.output_count; i++) {
            std::string blob_name;
            iss >> blob_name;
            info.output_blobs.push_back(blob_name);
        }

        // パラメータ
        std::string param;
        while (iss >> param) {
            auto eq_pos = param.find('=');
            if (eq_pos != std::string::npos) {
                int key = std::stoi(param.substr(0, eq_pos));
                std::string value = param.substr(eq_pos + 1);
                info.parameters[key] = value;
            }
        }

        return info;
    }

    static void print_layer_statistics(const std::map<std::string, int>& layer_counts) {
        std::cout << "Layer Type Statistics:" << std::endl;
        for (const auto& pair : layer_counts) {
            std::cout << "  " << std::setw(20) << pair.first << ": " << pair.second << std::endl;
        }
        std::cout << std::endl;
    }

    static void print_detailed_layer_info(const std::vector<LayerInfo>& layers) {
        std::cout << "Detailed Layer Information:" << std::endl;
        std::cout << std::setw(4) << "ID" << " "
                  << std::setw(15) << "Type" << " "
                  << std::setw(20) << "Name" << " "
                  << std::setw(10) << "Inputs" << " "
                  << std::setw(10) << "Outputs" << " "
                  << "Parameters" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (size_t i = 0; i < layers.size(); i++) {
            const LayerInfo& layer = layers[i];

            std::cout << std::setw(4) << i << " "
                      << std::setw(15) << layer.type << " "
                      << std::setw(20) << layer.name << " "
                      << std::setw(10) << layer.input_count << " "
                      << std::setw(10) << layer.output_count << " ";

            // 主要パラメータの表示
            if (layer.type == "Convolution") {
                print_conv_params(layer.parameters);
            } else if (layer.type == "InnerProduct") {
                print_fc_params(layer.parameters);
            } else if (layer.type == "Pooling") {
                print_pooling_params(layer.parameters);
            }

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    static void print_conv_params(const std::map<int, std::string>& params) {
        auto it = params.find(0);  // 出力チャンネル数
        if (it != params.end()) {
            std::cout << "out=" << it->second;
        }

        it = params.find(1);  // カーネルサイズ
        if (it != params.end()) {
            std::cout << " k=" << it->second;
        }

        it = params.find(2);  // ストライド
        if (it != params.end()) {
            std::cout << " s=" << it->second;
        }

        it = params.find(3);  // パディング
        if (it != params.end()) {
            std::cout << " p=" << it->second;
        }
    }

    static void print_fc_params(const std::map<int, std::string>& params) {
        auto it = params.find(0);  // 出力サイズ
        if (it != params.end()) {
            std::cout << "out=" << it->second;
        }

        it = params.find(1);  // バイアス項
        if (it != params.end()) {
            std::cout << " bias=" << it->second;
        }
    }

    static void print_pooling_params(const std::map<int, std::string>& params) {
        auto it = params.find(0);  // プーリングタイプ
        if (it != params.end()) {
            std::string type = (it->second == "0") ? "Max" : "Avg";
            std::cout << "type=" << type;
        }

        it = params.find(1);  // カーネルサイズ
        if (it != params.end()) {
            std::cout << " k=" << it->second;
        }

        it = params.find(2);  // ストライド
        if (it != params.end()) {
            std::cout << " s=" << it->second;
        }
    }

    static void analyze_model_complexity(const std::vector<LayerInfo>& layers) {
        std::cout << "Model Complexity Analysis:" << std::endl;

        long long total_params = 0;
        long long total_flops = 0;

        for (const auto& layer : layers) {
            if (layer.type == "Convolution") {
                auto params = estimate_conv_complexity(layer.parameters);
                total_params += params.first;
                total_flops += params.second;
            } else if (layer.type == "InnerProduct") {
                auto params = estimate_fc_complexity(layer.parameters);
                total_params += params.first;
                total_flops += params.second;
            }
        }

        std::cout << "  Estimated parameters: " << format_number(total_params) << std::endl;
        std::cout << "  Estimated FLOPs: " << format_number(total_flops) << std::endl;
    }

    static std::pair<long long, long long> estimate_conv_complexity(const std::map<int, std::string>& params) {
        // 簡略化された推定（実際の計算には入力サイズが必要）
        int out_channels = 0;
        int kernel_size = 1;

        auto it = params.find(0);
        if (it != params.end()) {
            out_channels = std::stoi(it->second);
        }

        it = params.find(1);
        if (it != params.end()) {
            kernel_size = std::stoi(it->second);
        }

        // 仮定: 入力チャンネル数=64, 出力サイズ=56x56
        long long weight_params = out_channels * 64 * kernel_size * kernel_size;
        long long flops = weight_params * 56 * 56;

        return {weight_params, flops};
    }

    static std::pair<long long, long long> estimate_fc_complexity(const std::map<int, std::string>& params) {
        int out_size = 0;

        auto it = params.find(0);
        if (it != params.end()) {
            out_size = std::stoi(it->second);
        }

        // 仮定: 入力サイズ=1024
        long long weight_params = 1024 * out_size;
        long long flops = weight_params;

        return {weight_params, flops};
    }

    static std::string format_number(long long num) {
        if (num >= 1000000000) {
            return std::to_string(num / 1000000000) + "G";
        } else if (num >= 1000000) {
            return std::to_string(num / 1000000) + "M";
        } else if (num >= 1000) {
            return std::to_string(num / 1000) + "K";
        } else {
            return std::to_string(num);
        }
    }
};
```

## 11.3 パフォーマンス問題の診断

### ボトルネック特定ツール

```cpp
class PerformanceProfiler {
private:
    struct LayerProfile {
        std::string name;
        std::string type;
        std::chrono::duration<double, std::milli> execution_time;
        size_t memory_usage;
        int call_count;
    };

    std::vector<LayerProfile> layer_profiles;
    std::chrono::high_resolution_clock::time_point start_time;
    bool profiling_enabled;

public:
    PerformanceProfiler() : profiling_enabled(false) {}

    void start_profiling() {
        profiling_enabled = true;
        layer_profiles.clear();
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Performance profiling started" << std::endl;
    }

    void stop_profiling() {
        profiling_enabled = false;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double, std::milli>(end_time - start_time);

        std::cout << "Performance profiling stopped" << std::endl;
        print_profiling_results(total_time);
    }

    ncnn::Mat profile_inference(ncnn::Net& net, const ncnn::Mat& input,
                              const std::string& input_name = "input",
                              const std::string& output_name = "output") {
        if (!profiling_enabled) {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_name.c_str(), input);

            ncnn::Mat output;
            ex.extract(output_name.c_str(), output);
            return output;
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        ncnn::Extractor ex = net.create_extractor();

        // 入力設定の時間測定
        auto input_start = std::chrono::high_resolution_clock::now();
        ex.input(input_name.c_str(), input);
        auto input_end = std::chrono::high_resolution_clock::now();

        record_operation("Input", "InputLayer", input_start, input_end, 0);

        // 推論実行の時間測定
        auto extract_start = std::chrono::high_resolution_clock::now();
        ncnn::Mat output;
        ex.extract(output_name.c_str(), output);
        auto extract_end = std::chrono::high_resolution_clock::now();

        record_operation("Extract", "OutputLayer", extract_start, extract_end, 0);

        auto inference_end = std::chrono::high_resolution_clock::now();
        auto total_inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start);

        std::cout << "Inference completed in " << total_inference_time.count() << " ms" << std::endl;

        return output;
    }

    void analyze_bottlenecks() {
        if (layer_profiles.empty()) {
            std::cout << "No profiling data available" << std::endl;
            return;
        }

        std::cout << "\n=== Bottleneck Analysis ===" << std::endl;

        // 実行時間でソート
        auto sorted_profiles = layer_profiles;
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                  [](const LayerProfile& a, const LayerProfile& b) {
                      return a.execution_time > b.execution_time;
                  });

        double total_time = 0.0;
        for (const auto& profile : sorted_profiles) {
            total_time += profile.execution_time.count();
        }

        std::cout << "Top 10 slowest operations:" << std::endl;
        std::cout << std::setw(25) << "Operation" << " "
                  << std::setw(15) << "Type" << " "
                  << std::setw(12) << "Time (ms)" << " "
                  << std::setw(10) << "% of Total" << " "
                  << std::setw(8) << "Calls" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (int i = 0; i < std::min(10, (int)sorted_profiles.size()); i++) {
            const auto& profile = sorted_profiles[i];
            double percentage = (profile.execution_time.count() / total_time) * 100.0;

            std::cout << std::setw(25) << profile.name << " "
                      << std::setw(15) << profile.type << " "
                      << std::setw(12) << std::fixed << std::setprecision(2) << profile.execution_time.count() << " "
                      << std::setw(10) << std::fixed << std::setprecision(1) << percentage << "%" << " "
                      << std::setw(8) << profile.call_count << std::endl;
        }

        // 最適化提案
        suggest_optimizations(sorted_profiles);
    }

    void benchmark_different_configurations() {
        std::cout << "\n=== Configuration Benchmark ===" << std::endl;

        std::vector<std::pair<std::string, ncnn::Option>> configs = {
            {"Default", create_default_option()},
            {"CPU Optimized", create_cpu_optimized_option()},
            {"Memory Optimized", create_memory_optimized_option()},
#if NCNN_VULKAN
            {"GPU", create_gpu_option()},
#endif
        };

        ncnn::Net net;
        // 実際のモデルを読み込む必要があります
        // net.load_param("model.param");
        // net.load_model("model.param", "model.bin");

        ncnn::Mat test_input(224, 224, 3);
        test_input.fill(0.5f);

        for (const auto& config : configs) {
            std::cout << "Testing configuration: " << config.first << std::endl;

            net.opt = config.second;

            auto start = std::chrono::high_resolution_clock::now();
            const int iterations = 10;

            for (int i = 0; i < iterations; i++) {
                ncnn::Extractor ex = net.create_extractor();
                ex.input("input", test_input);

                ncnn::Mat output;
                ex.extract("output", output);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto avg_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

            std::cout << "  Average time: " << avg_time << " ms" << std::endl;
        }
    }

private:
    void record_operation(const std::string& name, const std::string& type,
                         std::chrono::high_resolution_clock::time_point start,
                         std::chrono::high_resolution_clock::time_point end,
                         size_t memory) {
        LayerProfile profile;
        profile.name = name;
        profile.type = type;
        profile.execution_time = std::chrono::duration<double, std::milli>(end - start);
        profile.memory_usage = memory;
        profile.call_count = 1;

        // 既存のプロファイルを更新または新規追加
        auto it = std::find_if(layer_profiles.begin(), layer_profiles.end(),
                              [&name](const LayerProfile& p) { return p.name == name; });

        if (it != layer_profiles.end()) {
            it->execution_time += profile.execution_time;
            it->call_count++;
        } else {
            layer_profiles.push_back(profile);
        }
    }

    void print_profiling_results(std::chrono::duration<double, std::milli> total_time) {
        std::cout << "\n=== Profiling Results ===" << std::endl;
        std::cout << "Total execution time: " << total_time.count() << " ms" << std::endl;
        std::cout << "Number of operations: " << layer_profiles.size() << std::endl;

        double sum_layer_time = 0.0;
        for (const auto& profile : layer_profiles) {
            sum_layer_time += profile.execution_time.count();
        }

        std::cout << "Measured layer time: " << sum_layer_time << " ms" << std::endl;
        std::cout << "Overhead: " << (total_time.count() - sum_layer_time) << " ms" << std::endl;
    }

    void suggest_optimizations(const std::vector<LayerProfile>& profiles) {
        std::cout << "\n=== Optimization Suggestions ===" << std::endl;

        // ボトルネックレイヤーの分析
        if (!profiles.empty()) {
            const auto& slowest = profiles[0];
            double total_time = 0.0;
            for (const auto& p : profiles) {
                total_time += p.execution_time.count();
            }

            double slowest_percentage = (slowest.execution_time.count() / total_time) * 100.0;

            if (slowest_percentage > 50.0) {
                std::cout << "⚠ Major bottleneck detected in " << slowest.name << " (" << slowest_percentage << "% of total time)" << std::endl;

                if (slowest.type == "Convolution") {
                    std::cout << "Convolution optimization suggestions: "
                             << "Enable Winograd convolution for 3x3 kernels, try different thread counts, or consider model quantization" << std::endl;
                } else if (slowest.type == "InnerProduct") {
                    std::cout << "InnerProduct optimization suggestions: "
                             << "Enable SGEMM convolution or consider sparse weight matrices" << std::endl;
                }
            }

            if (slowest_percentage > 20.0) {
                std::cout << "Additional optimization options: "
                         << "Consider GPU acceleration with Vulkan, try different memory allocators, or experiment with different thread counts" << std::endl;
            }
        }
    }

    ncnn::Option create_default_option() {
        ncnn::Option opt;
        return opt;
    }

    ncnn::Option create_cpu_optimized_option() {
        ncnn::Option opt;
        opt.num_threads = std::thread::hardware_concurrency();
        opt.use_winograd_convolution = true;
        opt.use_sgemm_convolution = true;
        opt.use_int8_inference = false;
        return opt;
    }

    ncnn::Option create_memory_optimized_option() {
        ncnn::Option opt;
        opt.num_threads = 2;
        opt.use_memory_pool = true;
        opt.use_winograd_convolution = false;
        opt.use_sgemm_convolution = false;
        return opt;
    }

#if NCNN_VULKAN
    ncnn::Option create_gpu_option() {
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        return opt;
    }
#endif
};
```

これらのトラブルシューティング技術を活用することで、ncnn開発における問題を効率的に特定し、解決することができます。次章では、参考資料とコミュニティリソースについて紹介します。