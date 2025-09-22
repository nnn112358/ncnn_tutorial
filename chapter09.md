# 9. 実践的なアプリケーション開発

本章では、これまで学んだ技術を統合して、実用的なアプリケーションを開発する方法を学習します。リアルタイム画像処理、バッチ処理、エラーハンドリング、ログとデバッグについて詳しく解説します。

## 9.1 リアルタイム画像処理

リアルタイム画像処理アプリケーションでは、低遅延と安定した性能が求められます。Webカメラやビデオストリームからの入力に対してリアルタイムで推論を実行するシステムを実装します。

### リアルタイムビデオ処理基盤

```cpp
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

class RealTimeProcessor {
private:
    struct Frame {
        cv::Mat image;
        std::chrono::high_resolution_clock::time_point timestamp;
        int frame_id;
    };

    ncnn::Net net;
    std::queue<Frame> input_queue;
    std::queue<Frame> output_queue;
    std::mutex input_mutex, output_mutex;
    std::condition_variable input_cv, output_cv;

    std::thread capture_thread;
    std::thread inference_thread;
    std::thread display_thread;

    std::atomic<bool> running;
    std::atomic<int> frame_counter;

    // 性能統計
    std::atomic<double> avg_inference_time;
    std::atomic<double> fps;
    std::queue<double> inference_times;

    static constexpr int MAX_QUEUE_SIZE = 5;  // フレームドロップ防止

public:
    RealTimeProcessor() : running(false), frame_counter(0), avg_inference_time(0.0), fps(0.0) {
        // ncnn設定（リアルタイム用最適化）
        net.opt.use_vulkan_compute = true;
        net.opt.num_threads = 2;  // リアルタイム処理では少ないスレッド数が有効
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
    }

    bool initialize(const std::string& param_path, const std::string& bin_path) {
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load model for real-time processing" << std::endl;
            return false;
        }

        std::cout << "Real-time processor initialized successfully" << std::endl;
        return true;
    }

    void start_processing(int camera_id = 0) {
        running = true;

        // スレッドの開始
        capture_thread = std::thread(&RealTimeProcessor::capture_loop, this, camera_id);
        inference_thread = std::thread(&RealTimeProcessor::inference_loop, this);
        display_thread = std::thread(&RealTimeProcessor::display_loop, this);

        std::cout << "Real-time processing started. Press 'q' to quit." << std::endl;
    }

    void stop_processing() {
        running = false;

        // 全スレッドの終了を待機
        input_cv.notify_all();
        output_cv.notify_all();

        if (capture_thread.joinable()) capture_thread.join();
        if (inference_thread.joinable()) inference_thread.join();
        if (display_thread.joinable()) display_thread.join();

        std::cout << "Real-time processing stopped" << std::endl;
    }

    void print_performance_stats() {
        std::cout << "\n=== Real-time Performance Statistics ===" << std::endl;
        std::cout << "Average Inference Time: " << avg_inference_time.load() << " ms" << std::endl;
        std::cout << "Current FPS: " << fps.load() << std::endl;
        std::cout << "Processed Frames: " << frame_counter.load() << std::endl;
    }

private:
    void capture_loop(int camera_id) {
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera " << camera_id << std::endl;
            running = false;
            return;
        }

        // カメラ設定の最適化
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // バッファサイズを最小化

        cv::Mat frame;
        while (running) {
            if (!cap.read(frame) || frame.empty()) {
                continue;
            }

            Frame processed_frame;
            processed_frame.image = frame.clone();
            processed_frame.timestamp = std::chrono::high_resolution_clock::now();
            processed_frame.frame_id = frame_counter.fetch_add(1);

            // 入力キューに追加
            {
                std::unique_lock<std::mutex> lock(input_mutex);

                // キューサイズ制限（フレームドロップ）
                while (input_queue.size() >= MAX_QUEUE_SIZE && running) {
                    input_queue.pop();  // 古いフレームを破棄
                }

                input_queue.push(processed_frame);
            }
            input_cv.notify_one();

            // FPS計算（キャプチャベース）
            static auto last_time = std::chrono::high_resolution_clock::now();
            static int fps_counter = 0;
            fps_counter++;

            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time - last_time).count();

            if (elapsed >= 1.0) {  // 1秒ごとにFPS更新
                fps.store(fps_counter / elapsed);
                fps_counter = 0;
                last_time = current_time;
            }
        }
    }

    void inference_loop() {
        while (running) {
            Frame frame;

            // 入力フレームの取得
            {
                std::unique_lock<std::mutex> lock(input_mutex);
                input_cv.wait(lock, [this] { return !input_queue.empty() || !running; });

                if (!running) break;

                frame = input_queue.front();
                input_queue.pop();
            }

            // 推論実行
            auto inference_start = std::chrono::high_resolution_clock::now();

            Frame result_frame = process_frame(frame);

            auto inference_end = std::chrono::high_resolution_clock::now();
            double inference_time = std::chrono::duration<double, std::milli>(
                inference_end - inference_start).count();

            // 推論時間の統計更新
            update_inference_stats(inference_time);

            // 出力キューに追加
            {
                std::unique_lock<std::mutex> lock(output_mutex);

                // 出力キューサイズ制限
                while (output_queue.size() >= MAX_QUEUE_SIZE && running) {
                    output_queue.pop();
                }

                output_queue.push(result_frame);
            }
            output_cv.notify_one();
        }
    }

    void display_loop() {
        cv::namedWindow("Real-time Processing", cv::WINDOW_AUTOSIZE);

        while (running) {
            Frame frame;

            // 出力フレームの取得
            {
                std::unique_lock<std::mutex> lock(output_mutex);
                output_cv.wait(lock, [this] { return !output_queue.empty() || !running; });

                if (!running) break;

                frame = output_queue.front();
                output_queue.pop();
            }

            // 性能情報の描画
            draw_performance_info(frame.image);

            // 表示
            cv::imshow("Real-time Processing", frame.image);

            char key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q'またはESCキー
                running = false;
                break;
            }
        }

        cv::destroyAllWindows();
    }

    Frame process_frame(const Frame& input_frame) {
        Frame result_frame = input_frame;

        try {
            // 前処理
            ncnn::Mat ncnn_input = preprocess_image(input_frame.image);

            // 推論実行
            ncnn::Extractor ex = net.create_extractor();
            ex.input("input", ncnn_input);

            ncnn::Mat output;
            ex.extract("output", output);

            // 後処理と可視化
            result_frame.image = postprocess_and_visualize(input_frame.image, output);

        } catch (const std::exception& e) {
            std::cerr << "Error in frame processing: " << e.what() << std::endl;
            // エラー時は元のフレームをそのまま返す
        }

        return result_frame;
    }

    ncnn::Mat preprocess_image(const cv::Mat& image) {
        // 高速前処理（リアルタイム用に最適化）
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        // ncnn::Matに直接変換（コピーを最小化）
        ncnn::Mat input = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, 224, 224);

        // 正規化
        const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        input.substract_mean_normalize(0, norm_vals);

        return input;
    }

    cv::Mat postprocess_and_visualize(const cv::Mat& original, const ncnn::Mat& output) {
        cv::Mat result = original.clone();

        // 出力の解析と可視化
        // この部分は具体的なモデル（分類、検出等）に応じて実装

        return result;
    }

    void update_inference_stats(double inference_time) {
        static std::mutex stats_mutex;
        std::lock_guard<std::mutex> lock(stats_mutex);

        inference_times.push(inference_time);

        // 過去100フレームの平均を計算
        if (inference_times.size() > 100) {
            inference_times.pop();
        }

        double sum = 0.0;
        std::queue<double> temp_queue = inference_times;
        while (!temp_queue.empty()) {
            sum += temp_queue.front();
            temp_queue.pop();
        }

        avg_inference_time.store(sum / inference_times.size());
    }

    void draw_performance_info(cv::Mat& image) {
        // 性能情報の描画
        std::stringstream info;
        info << "FPS: " << std::fixed << std::setprecision(1) << fps.load();
        info << " | Inference: " << std::fixed << std::setprecision(1) << avg_inference_time.load() << "ms";
        info << " | Frames: " << frame_counter.load();

        cv::putText(image, info.str(), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 負荷インジケーター
        double load_ratio = avg_inference_time.load() / 33.33;  // 30FPS基準
        cv::Scalar load_color = load_ratio > 1.0 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        cv::rectangle(image, cv::Point(10, 50), cv::Point(10 + (int)(200 * std::min(1.0, load_ratio)), 70),
                     load_color, cv::FILLED);
        cv::rectangle(image, cv::Point(10, 50), cv::Point(210, 70), cv::Scalar(255, 255, 255), 2);
    }
};
```

### ライブストリーミング対応

```cpp
class LiveStreamProcessor : public RealTimeProcessor {
private:
    cv::VideoWriter video_writer;
    bool recording;
    std::string output_filename;

public:
    LiveStreamProcessor() : recording(false) {}

    void start_recording(const std::string& filename, int fourcc = cv::VideoWriter::fourcc('X','V','I','D')) {
        output_filename = filename;
        video_writer.open(filename, fourcc, 30.0, cv::Size(640, 480));

        if (video_writer.isOpened()) {
            recording = true;
            std::cout << "Recording started: " << filename << std::endl;
        } else {
            std::cerr << "Failed to start recording" << std::endl;
        }
    }

    void stop_recording() {
        if (recording) {
            recording = false;
            video_writer.release();
            std::cout << "Recording stopped: " << output_filename << std::endl;
        }
    }

protected:
    void display_loop() override {
        cv::namedWindow("Live Stream Processing", cv::WINDOW_AUTOSIZE);

        while (running) {
            Frame frame;

            {
                std::unique_lock<std::mutex> lock(output_mutex);
                output_cv.wait(lock, [this] { return !output_queue.empty() || !running; });

                if (!running) break;

                frame = output_queue.front();
                output_queue.pop();
            }

            draw_performance_info(frame.image);

            // 録画
            if (recording && video_writer.isOpened()) {
                video_writer.write(frame.image);
            }

            cv::imshow("Live Stream Processing", frame.image);

            char key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                running = false;
                break;
            } else if (key == 'r') {
                if (!recording) {
                    start_recording("output_" + get_timestamp() + ".avi");
                } else {
                    stop_recording();
                }
            }
        }

        if (recording) {
            stop_recording();
        }

        cv::destroyAllWindows();
    }

private:
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }
};
```

## 9.2 バッチ処理

大量の画像を効率的に処理するバッチ処理システムを実装します。

```cpp
#include <filesystem>
#include <algorithm>
#include <execution>

class BatchProcessor {
private:
    ncnn::Net net;
    std::string input_directory;
    std::string output_directory;
    std::vector<std::string> supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"};

    // 進捗管理
    std::atomic<int> processed_count;
    std::atomic<int> total_count;
    std::atomic<bool> processing;

public:
    struct BatchConfig {
        int batch_size = 16;
        int num_workers = 4;
        bool save_results = true;
        bool show_progress = true;
        std::string result_format = "json";  // json, csv, xml
    };

    BatchProcessor() : processed_count(0), total_count(0), processing(false) {
        // バッチ処理用最適化
        net.opt.use_vulkan_compute = true;
        net.opt.num_threads = 1;  // ワーカー毎に1スレッド
        net.opt.use_memory_pool = true;
    }

    bool initialize(const std::string& param_path, const std::string& bin_path,
                   const std::string& input_dir, const std::string& output_dir) {
        // モデル読み込み
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load model for batch processing" << std::endl;
            return false;
        }

        input_directory = input_dir;
        output_directory = output_dir;

        // 出力ディレクトリの作成
        std::filesystem::create_directories(output_directory);

        std::cout << "Batch processor initialized" << std::endl;
        return true;
    }

    void process_batch(const BatchConfig& config = BatchConfig()) {
        // 入力ファイルリストの取得
        std::vector<std::string> image_files = get_image_files();

        if (image_files.empty()) {
            std::cout << "No image files found in " << input_directory << std::endl;
            return;
        }

        total_count = image_files.size();
        processed_count = 0;
        processing = true;

        std::cout << "Starting batch processing of " << total_count << " images..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // 並列処理の実行
        if (config.num_workers > 1) {
            process_parallel(image_files, config);
        } else {
            process_sequential(image_files, config);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time);

        processing = false;

        std::cout << "\nBatch processing completed!" << std::endl;
        std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
        std::cout << "Average time per image: " << (duration.count() / total_count) << " seconds" << std::endl;
        std::cout << "Throughput: " << (total_count / duration.count()) << " images/second" << std::endl;
    }

    void start_progress_monitor() {
        std::thread monitor_thread([this]() {
            while (processing) {
                int current = processed_count.load();
                int total = total_count.load();

                if (total > 0) {
                    double progress = (double)current / total * 100.0;
                    std::cout << "\rProgress: " << current << "/" << total
                             << " (" << std::fixed << std::setprecision(1) << progress << "%)";
                    std::cout.flush();
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });

        monitor_thread.detach();
    }

private:
    std::vector<std::string> get_image_files() {
        std::vector<std::string> files;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(input_directory)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (std::find(supported_extensions.begin(), supported_extensions.end(), extension)
                    != supported_extensions.end()) {
                    files.push_back(entry.path().string());
                }
            }
        }

        std::sort(files.begin(), files.end());
        return files;
    }

    void process_sequential(const std::vector<std::string>& files, const BatchConfig& config) {
        if (config.show_progress) {
            start_progress_monitor();
        }

        for (const auto& file : files) {
            process_single_image(file, config);
            processed_count.fetch_add(1);
        }
    }

    void process_parallel(const std::vector<std::string>& files, const BatchConfig& config) {
        if (config.show_progress) {
            start_progress_monitor();
        }

        // ワーカープールの作成
        std::vector<std::thread> workers;
        std::queue<std::string> file_queue;
        std::mutex queue_mutex;

        // ファイルキューの初期化
        for (const auto& file : files) {
            file_queue.push(file);
        }

        // ワーカーの開始
        for (int i = 0; i < config.num_workers; i++) {
            workers.emplace_back([this, &file_queue, &queue_mutex, &config]() {
                // 各ワーカー用のネットワークをコピー
                ncnn::Net worker_net;
                worker_net.opt = net.opt;
                // 注意: 実際の実装では、各ワーカーが独自のncnn::Netインスタンスを持つ必要があります

                while (true) {
                    std::string file;

                    {
                        std::lock_guard<std::mutex> lock(queue_mutex);
                        if (file_queue.empty()) {
                            break;
                        }
                        file = file_queue.front();
                        file_queue.pop();
                    }

                    process_single_image(file, config);
                    processed_count.fetch_add(1);
                }
            });
        }

        // 全ワーカーの完了を待機
        for (auto& worker : workers) {
            worker.join();
        }
    }

    void process_single_image(const std::string& image_path, const BatchConfig& config) {
        try {
            // 画像読み込み
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                return;
            }

            // 前処理
            ncnn::Mat input = preprocess_image(image);

            // 推論実行
            ncnn::Extractor ex = net.create_extractor();
            ex.input("input", input);

            ncnn::Mat output;
            ex.extract("output", output);

            // 結果の保存
            if (config.save_results) {
                save_result(image_path, output, config.result_format);
            }

        } catch (const std::exception& e) {
            std::cerr << "Error processing " << image_path << ": " << e.what() << std::endl;
        }
    }

    ncnn::Mat preprocess_image(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(224, 224));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        ncnn::Mat input = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, 224, 224);

        const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        input.substract_mean_normalize(0, norm_vals);

        return input;
    }

    void save_result(const std::string& image_path, const ncnn::Mat& output, const std::string& format) {
        std::filesystem::path input_path(image_path);
        std::string output_filename = input_path.stem().string() + "_result." + format;
        std::string output_path = output_directory + "/" + output_filename;

        if (format == "json") {
            save_result_json(output_path, output);
        } else if (format == "csv") {
            save_result_csv(output_path, output);
        }
    }

    void save_result_json(const std::string& output_path, const ncnn::Mat& output) {
        std::ofstream file(output_path);
        file << "{\n";
        file << "  \"predictions\": [\n";

        const float* data = (const float*)output.data;
        for (int i = 0; i < output.w; i++) {
            file << "    " << data[i];
            if (i < output.w - 1) file << ",";
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";
    }

    void save_result_csv(const std::string& output_path, const ncnn::Mat& output) {
        std::ofstream file(output_path);
        file << "class_id,score\n";

        const float* data = (const float*)output.data;
        for (int i = 0; i < output.w; i++) {
            file << i << "," << data[i] << "\n";
        }
    }
};
```

## 9.3 エラーハンドリング

堅牢なアプリケーションのための包括的なエラーハンドリングシステムを実装します。

```cpp
#include <exception>
#include <stdexcept>

enum class NCNNErrorCode {
    SUCCESS = 0,
    MODEL_LOAD_FAILED,
    INVALID_INPUT,
    INFERENCE_FAILED,
    MEMORY_ALLOCATION_FAILED,
    GPU_ERROR,
    INVALID_CONFIGURATION,
    TIMEOUT,
    UNKNOWN_ERROR
};

class NCNNException : public std::exception {
private:
    NCNNErrorCode error_code;
    std::string message;
    std::string context;

public:
    NCNNException(NCNNErrorCode code, const std::string& msg, const std::string& ctx = "")
        : error_code(code), message(msg), context(ctx) {}

    const char* what() const noexcept override {
        static std::string full_message;
        full_message = "[" + error_code_to_string(error_code) + "] " + message;
        if (!context.empty()) {
            full_message += " (Context: " + context + ")";
        }
        return full_message.c_str();
    }

    NCNNErrorCode get_error_code() const { return error_code; }
    const std::string& get_context() const { return context; }

private:
    std::string error_code_to_string(NCNNErrorCode code) const {
        switch (code) {
            case NCNNErrorCode::SUCCESS: return "SUCCESS";
            case NCNNErrorCode::MODEL_LOAD_FAILED: return "MODEL_LOAD_FAILED";
            case NCNNErrorCode::INVALID_INPUT: return "INVALID_INPUT";
            case NCNNErrorCode::INFERENCE_FAILED: return "INFERENCE_FAILED";
            case NCNNErrorCode::MEMORY_ALLOCATION_FAILED: return "MEMORY_ALLOCATION_FAILED";
            case NCNNErrorCode::GPU_ERROR: return "GPU_ERROR";
            case NCNNErrorCode::INVALID_CONFIGURATION: return "INVALID_CONFIGURATION";
            case NCNNErrorCode::TIMEOUT: return "TIMEOUT";
            default: return "UNKNOWN_ERROR";
        }
    }
};

class RobustInferenceEngine {
private:
    ncnn::Net net;
    bool model_loaded;
    std::chrono::milliseconds timeout_duration;
    int max_retry_count;

public:
    RobustInferenceEngine(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000),
                         int max_retries = 3)
        : model_loaded(false), timeout_duration(timeout), max_retry_count(max_retries) {}

    void load_model(const std::string& param_path, const std::string& bin_path) {
        try {
            if (!std::filesystem::exists(param_path)) {
                throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                                  "Parameter file not found: " + param_path);
            }

            if (!std::filesystem::exists(bin_path)) {
                throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                                  "Binary file not found: " + bin_path);
            }

            int ret1 = net.load_param(param_path.c_str());
            int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

            if (ret1 != 0) {
                throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                                  "Failed to load parameter file",
                                  "Return code: " + std::to_string(ret1));
            }

            if (ret2 != 0) {
                throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                                  "Failed to load binary file",
                                  "Return code: " + std::to_string(ret2));
            }

            model_loaded = true;
            validate_model();

        } catch (const NCNNException&) {
            throw;  // NCNNExceptionは再スロー
        } catch (const std::exception& e) {
            throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                              "Unexpected error during model loading: " + std::string(e.what()));
        }
    }

    ncnn::Mat safe_inference(const ncnn::Mat& input, const std::string& input_name = "input",
                           const std::string& output_name = "output") {
        if (!model_loaded) {
            throw NCNNException(NCNNErrorCode::INVALID_CONFIGURATION, "Model not loaded");
        }

        validate_input(input);

        int retry_count = 0;
        while (retry_count <= max_retry_count) {
            try {
                return execute_inference_with_timeout(input, input_name, output_name);

            } catch (const NCNNException& e) {
                if (e.get_error_code() == NCNNErrorCode::MEMORY_ALLOCATION_FAILED ||
                    e.get_error_code() == NCNNErrorCode::GPU_ERROR) {

                    retry_count++;
                    if (retry_count <= max_retry_count) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100 * retry_count));
                        continue;
                    }
                }
                throw;
            }
        }

        throw NCNNException(NCNNErrorCode::INFERENCE_FAILED,
                          "Maximum retry count exceeded");
    }

private:
    void validate_model() {
        // 簡単な推論テストでモデルの有効性を確認
        try {
            ncnn::Mat test_input(224, 224, 3);
            test_input.fill(0.5f);

            ncnn::Extractor ex = net.create_extractor();
            ex.input("input", test_input);

            ncnn::Mat test_output;
            int ret = ex.extract("output", test_output);

            if (ret != 0) {
                throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                                  "Model validation failed",
                                  "Extract return code: " + std::to_string(ret));
            }

        } catch (const NCNNException&) {
            throw;
        } catch (const std::exception& e) {
            throw NCNNException(NCNNErrorCode::MODEL_LOAD_FAILED,
                              "Model validation error: " + std::string(e.what()));
        }
    }

    void validate_input(const ncnn::Mat& input) {
        if (input.empty()) {
            throw NCNNException(NCNNErrorCode::INVALID_INPUT, "Input tensor is empty");
        }

        if (input.dims != 3 || input.w <= 0 || input.h <= 0 || input.c <= 0) {
            throw NCNNException(NCNNErrorCode::INVALID_INPUT,
                              "Invalid input dimensions: [" +
                              std::to_string(input.c) + ", " +
                              std::to_string(input.h) + ", " +
                              std::to_string(input.w) + "]");
        }

        // 入力値の範囲チェック
        const float* data = (const float*)input.data;
        size_t total_elements = input.c * input.h * input.w;

        for (size_t i = 0; i < total_elements; i++) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                throw NCNNException(NCNNErrorCode::INVALID_INPUT,
                                  "Input contains NaN or Inf values",
                                  "Element index: " + std::to_string(i));
            }
        }
    }

    ncnn::Mat execute_inference_with_timeout(const ncnn::Mat& input,
                                           const std::string& input_name,
                                           const std::string& output_name) {
        std::promise<ncnn::Mat> result_promise;
        std::future<ncnn::Mat> result_future = result_promise.get_future();

        std::thread inference_thread([&]() {
            try {
                ncnn::Extractor ex = net.create_extractor();

                int input_ret = ex.input(input_name.c_str(), input);
                if (input_ret != 0) {
                    throw NCNNException(NCNNErrorCode::INFERENCE_FAILED,
                                      "Failed to set input",
                                      "Input name: " + input_name + ", Return code: " + std::to_string(input_ret));
                }

                ncnn::Mat output;
                int extract_ret = ex.extract(output_name.c_str(), output);
                if (extract_ret != 0) {
                    throw NCNNException(NCNNErrorCode::INFERENCE_FAILED,
                                      "Failed to extract output",
                                      "Output name: " + output_name + ", Return code: " + std::to_string(extract_ret));
                }

                result_promise.set_value(output);

            } catch (...) {
                result_promise.set_exception(std::current_exception());
            }
        });

        if (result_future.wait_for(timeout_duration) == std::future_status::timeout) {
            // タイムアウトの場合、スレッドをデタッチ（強制終了は危険）
            inference_thread.detach();
            throw NCNNException(NCNNErrorCode::TIMEOUT,
                              "Inference timeout",
                              "Timeout: " + std::to_string(timeout_duration.count()) + "ms");
        }

        inference_thread.join();
        return result_future.get();
    }
};
```

## 9.4 ログとデバッグ

包括的なログシステムとデバッグ機能を実装します。

```cpp
#include <fstream>
#include <sstream>
#include <iomanip>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
private:
    LogLevel min_level;
    std::ofstream log_file;
    std::mutex log_mutex;
    bool console_output;
    bool file_output;

public:
    Logger(LogLevel level = LogLevel::INFO, bool console = true, bool file = false)
        : min_level(level), console_output(console), file_output(file) {

        if (file_output) {
            std::string filename = "ncnn_log_" + get_timestamp() + ".txt";
            log_file.open(filename, std::ios::app);
        }
    }

    ~Logger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    template<typename... Args>
    void log(LogLevel level, const std::string& format, Args... args) {
        if (level < min_level) {
            return;
        }

        std::string message = format_string(format, args...);
        std::string log_entry = create_log_entry(level, message);

        std::lock_guard<std::mutex> lock(log_mutex);

        if (console_output) {
            if (level >= LogLevel::ERROR) {
                std::cerr << log_entry << std::endl;
            } else {
                std::cout << log_entry << std::endl;
            }
        }

        if (file_output && log_file.is_open()) {
            log_file << log_entry << std::endl;
            log_file.flush();
        }
    }

    template<typename... Args>
    void debug(const std::string& format, Args... args) {
        log(LogLevel::DEBUG, format, args...);
    }

    template<typename... Args>
    void info(const std::string& format, Args... args) {
        log(LogLevel::INFO, format, args...);
    }

    template<typename... Args>
    void warning(const std::string& format, Args... args) {
        log(LogLevel::WARNING, format, args...);
    }

    template<typename... Args>
    void error(const std::string& format, Args... args) {
        log(LogLevel::ERROR, format, args...);
    }

    template<typename... Args>
    void critical(const std::string& format, Args... args) {
        log(LogLevel::CRITICAL, format, args...);
    }

private:
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }

    std::string create_log_entry(LogLevel level, const std::string& message) {
        std::stringstream ss;
        ss << "[" << get_timestamp() << "] "
           << "[" << level_to_string(level) << "] "
           << message;
        return ss.str();
    }

    template<typename... Args>
    std::string format_string(const std::string& format, Args... args) {
        size_t size = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, format.c_str(), args...);
        return std::string(buf.get(), buf.get() + size - 1);
    }
};

// グローバルロガーインスタンス
static Logger global_logger(LogLevel::INFO, true, true);

#define LOG_DEBUG(...) global_logger.debug(__VA_ARGS__)
#define LOG_INFO(...) global_logger.info(__VA_ARGS__)
#define LOG_WARNING(...) global_logger.warning(__VA_ARGS__)
#define LOG_ERROR(...) global_logger.error(__VA_ARGS__)
#define LOG_CRITICAL(...) global_logger.critical(__VA_ARGS__)

class DebugInferenceEngine {
private:
    RobustInferenceEngine engine;
    Logger debug_logger;

public:
    DebugInferenceEngine() : debug_logger(LogLevel::DEBUG, true, true) {}

    void load_model_with_debug(const std::string& param_path, const std::string& bin_path) {
        LOG_INFO("Loading model: param=%s, bin=%s", param_path.c_str(), bin_path.c_str());

        try {
            auto start = std::chrono::high_resolution_clock::now();
            engine.load_model(param_path, bin_path);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration<double, std::milli>(end - start);
            LOG_INFO("Model loaded successfully in %.2f ms", duration.count());

        } catch (const NCNNException& e) {
            LOG_ERROR("Failed to load model: %s", e.what());
            throw;
        }
    }

    ncnn::Mat inference_with_debug(const ncnn::Mat& input, const std::string& input_name = "input",
                                 const std::string& output_name = "output") {
        LOG_DEBUG("Starting inference: input_shape=[%d,%d,%d], input_name=%s, output_name=%s",
                 input.c, input.h, input.w, input_name.c_str(), output_name.c_str());

        try {
            auto start = std::chrono::high_resolution_clock::now();

            ncnn::Mat output = engine.safe_inference(input, input_name, output_name);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::milli>(end - start);

            LOG_DEBUG("Inference completed in %.2f ms, output_shape=[%d,%d,%d]",
                     duration.count(), output.c, output.h, output.w);

            return output;

        } catch (const NCNNException& e) {
            LOG_ERROR("Inference failed: %s", e.what());
            throw;
        }
    }

    void dump_tensor_stats(const ncnn::Mat& tensor, const std::string& name) {
        if (tensor.empty()) {
            LOG_DEBUG("Tensor %s is empty", name.c_str());
            return;
        }

        const float* data = (const float*)tensor.data;
        size_t total_elements = tensor.c * tensor.h * tensor.w;

        float min_val = *std::min_element(data, data + total_elements);
        float max_val = *std::max_element(data, data + total_elements);

        double sum = 0.0;
        for (size_t i = 0; i < total_elements; i++) {
            sum += data[i];
        }
        double mean = sum / total_elements;

        double variance = 0.0;
        for (size_t i = 0; i < total_elements; i++) {
            variance += (data[i] - mean) * (data[i] - mean);
        }
        variance /= total_elements;
        double std_dev = std::sqrt(variance);

        LOG_DEBUG("Tensor %s stats: shape=[%d,%d,%d], min=%.6f, max=%.6f, mean=%.6f, std=%.6f",
                 name.c_str(), tensor.c, tensor.h, tensor.w, min_val, max_val, mean, std_dev);
    }
};
```

これらの実装により、堅牢で実用的なncnnアプリケーションを開発することができます。次章では、さらに高度なトピックについて学習します。