# 6. 画像分類の実装

画像分類は、コンピュータビジョンにおける最も基本的で重要なタスクの一つです。本章では、ncnnを使用して実用的な画像分類アプリケーションを実装する方法を詳しく解説します。

## 6.1 画像前処理

ディープラーニングモデルに画像を入力する前に、適切な前処理を行う必要があります。この前処理は、モデルの学習時に使用されたものと同じである必要があります。

### 6.1.1 画像読み込み

まず、様々な形式の画像ファイルを読み込む方法を実装します。

**OpenCVを使用した画像読み込み**:

```cpp
#include <opencv2/opencv.hpp>
#include <ncnn/mat.h>

class ImageLoader {
public:
    static cv::Mat load_image(const std::string& image_path) {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }
        return image;
    }

    static cv::Mat load_image_rgb(const std::string& image_path) {
        cv::Mat image = load_image(image_path);
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        return rgb_image;
    }
};
```

**OpenCVを使わない軽量な実装**（STB image使用）:

```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class LightImageLoader {
public:
    struct Image {
        unsigned char* data;
        int width;
        int height;
        int channels;

        ~Image() {
            if (data) {
                stbi_image_free(data);
            }
        }
    };

    static Image load_image(const std::string& image_path) {
        Image img;
        img.data = stbi_load(image_path.c_str(), &img.width, &img.height, &img.channels, 3);
        if (!img.data) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }
        return img;
    }
};
```

### 6.1.2 リサイズと正規化

ほとんどの画像分類モデルは、固定サイズの入力（例：224x224）を期待します。

```cpp
class ImagePreprocessor {
private:
    int target_width;
    int target_height;
    std::vector<float> mean_values;
    std::vector<float> std_values;
    bool normalize_to_01;

public:
    ImagePreprocessor(int width = 224, int height = 224,
                     const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                     const std::vector<float>& std = {0.229f, 0.224f, 0.225f},
                     bool normalize = true)
        : target_width(width), target_height(height),
          mean_values(mean), std_values(std), normalize_to_01(normalize) {}

    cv::Mat resize_image(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        return resized;
    }

    cv::Mat center_crop(const cv::Mat& image, int crop_size) {
        int h = image.rows;
        int w = image.cols;

        int start_x = (w - crop_size) / 2;
        int start_y = (h - crop_size) / 2;

        cv::Rect crop_rect(start_x, start_y, crop_size, crop_size);
        return image(crop_rect);
    }

    cv::Mat normalize_image(const cv::Mat& image) {
        cv::Mat normalized;
        image.convertTo(normalized, CV_32F);

        if (normalize_to_01) {
            normalized /= 255.0f;
        }

        // チャンネルごとの正規化
        std::vector<cv::Mat> channels;
        cv::split(normalized, channels);

        for (int i = 0; i < channels.size(); i++) {
            if (i < mean_values.size() && i < std_values.size()) {
                channels[i] = (channels[i] - mean_values[i]) / std_values[i];
            }
        }

        cv::merge(channels, normalized);
        return normalized;
    }

    // 完全な前処理パイプライン
    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat processed = image.clone();

        // リサイズ
        processed = resize_image(processed);

        // 正規化
        processed = normalize_image(processed);

        return processed;
    }
};
```

### 6.1.3 ncnn::Matへの変換

OpenCVのcv::Matからncnn::Matへの変換を行います。

```cpp
class MatConverter {
public:
    static ncnn::Mat cv_mat_to_ncnn_mat(const cv::Mat& cv_mat) {
        // OpenCVはHWC形式、ncnnはCHW形式
        int h = cv_mat.rows;
        int w = cv_mat.cols;
        int c = cv_mat.channels();

        ncnn::Mat ncnn_mat(w, h, c);

        if (cv_mat.type() == CV_32F) {
            // float型の場合
            const float* cv_data = (const float*)cv_mat.data;

            for (int ch = 0; ch < c; ch++) {
                float* ncnn_channel = ncnn_mat.channel(ch);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        ncnn_channel[y * w + x] = cv_data[(y * w + x) * c + ch];
                    }
                }
            }
        } else if (cv_mat.type() == CV_8UC3) {
            // uchar型の場合（正規化も同時に実行）
            const unsigned char* cv_data = cv_mat.data;

            for (int ch = 0; ch < c; ch++) {
                float* ncnn_channel = ncnn_mat.channel(ch);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        ncnn_channel[y * w + x] = (float)cv_data[(y * w + x) * c + ch] / 255.0f;
                    }
                }
            }
        }

        return ncnn_mat;
    }

    // より効率的な変換（OpenCVのchannels分離を使用）
    static ncnn::Mat cv_mat_to_ncnn_mat_optimized(const cv::Mat& cv_mat) {
        cv::Mat float_mat;
        cv_mat.convertTo(float_mat, CV_32F, 1.0/255.0);

        std::vector<cv::Mat> channels;
        cv::split(float_mat, channels);

        int h = cv_mat.rows;
        int w = cv_mat.cols;
        int c = cv_mat.channels();

        ncnn::Mat ncnn_mat(w, h, c);

        for (int ch = 0; ch < c; ch++) {
            memcpy(ncnn_mat.channel(ch), channels[ch].data, h * w * sizeof(float));
        }

        return ncnn_mat;
    }
};
```

## 6.2 モデルの読み込みと推論

画像分類モデルの読み込みと推論実行を管理するクラスを実装します。

```cpp
#include <ncnn/net.h>
#include <fstream>
#include <sstream>

class ImageClassifier {
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    ImagePreprocessor preprocessor;
    bool model_loaded;

public:
    ImageClassifier(int input_size = 224)
        : preprocessor(input_size, input_size), model_loaded(false) {
        // デフォルト設定
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
    }

    bool load_model(const std::string& param_path, const std::string& bin_path) {
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load model" << std::endl;
            return false;
        }

        model_loaded = true;
        std::cout << "Model loaded successfully" << std::endl;
        return true;
    }

    bool load_class_names(const std::string& class_file) {
        std::ifstream file(class_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open class file: " << class_file << std::endl;
            return false;
        }

        class_names.clear();
        std::string line;
        while (std::getline(file, line)) {
            // 改行文字を除去
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            class_names.push_back(line);
        }

        std::cout << "Loaded " << class_names.size() << " class names" << std::endl;
        return true;
    }

    struct ClassificationResult {
        int class_id;
        float score;
        std::string class_name;
    };

    std::vector<ClassificationResult> classify_image(const cv::Mat& image, int top_k = 5) {
        if (!model_loaded) {
            throw std::runtime_error("Model not loaded");
        }

        // 前処理
        cv::Mat processed = preprocessor.preprocess(image);
        ncnn::Mat input = MatConverter::cv_mat_to_ncnn_mat_optimized(processed);

        // 推論実行
        ncnn::Extractor ex = net.create_extractor();
        ex.input("data", input);  // 入力レイヤー名は実際のモデルに合わせて調整

        ncnn::Mat output;
        ex.extract("prob", output);  // 出力レイヤー名は実際のモデルに合わせて調整

        // Softmax適用（必要に応じて）
        apply_softmax(output);

        // 結果の解析
        return parse_classification_results(output, top_k);
    }

private:
    void apply_softmax(ncnn::Mat& mat) {
        float* data = (float*)mat.data;
        int size = mat.w;

        // 最大値を見つける（数値安定性のため）
        float max_val = *std::max_element(data, data + size);

        // Softmax計算
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            data[i] = std::exp(data[i] - max_val);
            sum += data[i];
        }

        for (int i = 0; i < size; i++) {
            data[i] /= sum;
        }
    }

    std::vector<ClassificationResult> parse_classification_results(const ncnn::Mat& output, int top_k) {
        const float* prob_data = (const float*)output.data;
        int num_classes = output.w;

        // スコアとインデックスのペアを作成
        std::vector<std::pair<float, int>> scores;
        for (int i = 0; i < num_classes; i++) {
            scores.push_back({prob_data[i], i});
        }

        // 降順でソート
        std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, int>>());

        // 上位top_k個の結果を作成
        std::vector<ClassificationResult> results;
        for (int i = 0; i < std::min(top_k, (int)scores.size()); i++) {
            ClassificationResult result;
            result.score = scores[i].first;
            result.class_id = scores[i].second;

            if (result.class_id < class_names.size()) {
                result.class_name = class_names[result.class_id];
            } else {
                result.class_name = "Unknown";
            }

            results.push_back(result);
        }

        return results;
    }
};
```

## 6.3 結果の後処理

分類結果の可視化と分析のための機能を実装します。

```cpp
class ResultVisualizer {
public:
    static void print_results(const std::vector<ImageClassifier::ClassificationResult>& results) {
        std::cout << "\nClassification Results:" << std::endl;
        std::cout << "-------------------------" << std::endl;

        for (size_t i = 0; i < results.size(); i++) {
            printf("%zu. %s (%.4f)\n",
                   i + 1,
                   results[i].class_name.c_str(),
                   results[i].score);
        }
    }

    static void draw_results_on_image(cv::Mat& image,
                                    const std::vector<ImageClassifier::ClassificationResult>& results,
                                    int max_results = 3) {
        // 画像の上部に結果を描画
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.6;
        int thickness = 2;
        cv::Scalar text_color(0, 255, 0);  // 緑色
        cv::Scalar bg_color(0, 0, 0);      // 黒色背景

        int y_offset = 30;
        for (int i = 0; i < std::min(max_results, (int)results.size()); i++) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3);
            ss << results[i].class_name << " (" << results[i].score << ")";
            std::string text = ss.str();

            // テキストサイズを取得
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

            // 背景矩形を描画
            cv::Point text_org(10, y_offset);
            cv::rectangle(image,
                         cv::Point(text_org.x, text_org.y - text_size.height),
                         cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                         bg_color, cv::FILLED);

            // テキストを描画
            cv::putText(image, text, text_org, font_face, font_scale, text_color, thickness);

            y_offset += text_size.height + baseline + 5;
        }
    }

    static void save_results_to_file(const std::vector<ImageClassifier::ClassificationResult>& results,
                                   const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return;
        }

        file << "class_id,class_name,score\n";
        for (const auto& result : results) {
            file << result.class_id << "," << result.class_name << "," << result.score << "\n";
        }

        std::cout << "Results saved to: " << output_file << std::endl;
    }
};
```

## 6.4 完全なサンプルコード

すべての要素を統合した完全な画像分類アプリケーションを示します。

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <ncnn/mat.h>

// 前述のクラス定義をここに含める
// - ImageLoader
// - ImagePreprocessor
// - MatConverter
// - ImageClassifier
// - ResultVisualizer

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model.param> <model.bin> <image.jpg>" << std::endl;
        return -1;
    }

    std::string param_file = argv[1];
    std::string bin_file = argv[2];
    std::string image_file = argv[3];

    try {
        // 分類器の初期化
        ImageClassifier classifier(224);  // 224x224入力サイズ

        // モデルの読み込み
        if (!classifier.load_model(param_file, bin_file)) {
            return -1;
        }

        // クラス名の読み込み（オプション）
        classifier.load_class_names("imagenet_classes.txt");

        // 画像の読み込み
        cv::Mat image = ImageLoader::load_image_rgb(image_file);
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;

        // 推論実行
        auto start_time = std::chrono::high_resolution_clock::now();

        auto results = classifier.classify_image(image, 5);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

        // 結果の表示
        ResultVisualizer::print_results(results);

        // 結果を画像に描画
        cv::Mat display_image = image.clone();
        ResultVisualizer::draw_results_on_image(display_image, results);

        // 結果画像の保存
        cv::cvtColor(display_image, display_image, cv::COLOR_RGB2BGR);
        cv::imwrite("result.jpg", display_image);
        std::cout << "Result image saved as result.jpg" << std::endl;

        // CSVファイルに結果を保存
        ResultVisualizer::save_results_to_file(results, "results.csv");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```

### コンパイル用のCMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(image_classification)

set(CMAKE_CXX_STANDARD 14)

# 必要なライブラリを検索
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

# 実行ファイルを作成
add_executable(image_classification
    main.cpp
)

# ライブラリをリンク
target_link_libraries(image_classification
    ncnn
    ${OpenCV_LIBS}
)

target_include_directories(image_classification PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
```

### 使用例

```bash
# ビルド
mkdir build && cd build
cmake ..
make -j4

# 実行
./image_classification mobilenet_v2.param mobilenet_v2.bin test_image.jpg
```

**期待される出力**:
```
Model loaded successfully
Loaded 1000 class names
Image loaded: 640x480
Inference time: 45 ms

Classification Results:
-------------------------
1. Egyptian cat (0.8234)
2. tabby, tabby cat (0.1456)
3. tiger cat (0.0234)
4. lynx, catamount (0.0045)
5. Persian cat (0.0023)

Result image saved as result.jpg
Results saved to: results.csv
```

この実装により、実用的な画像分類アプリケーションが完成しました。次章では、より複雑な物体検出タスクの実装について学習します。

### パフォーマンス最適化のヒント

実用的な画像分類アプリケーションでは、パフォーマンスの最適化が重要な要素となります。特に、リアルタイム処理や大量の画像処理が必要な場合には、系統的な最適化手法を適用することで、大幅な性能向上を実現できます。

**前処理の最適化**については、同じ画像に対して繰り返し前処理が実行されることを防ぐためのキャッシュ機能が有効です。特に、数百枚以上の画像を処理する場合や、動画のフレーム処理などでは、この最適化による効果が顕著に現れます：

```cpp
// 重複する前処理を避ける
static cv::Mat preprocessed_cache;
static std::string cached_image_path;

if (image_path != cached_image_path) {
    preprocessed_cache = preprocessor.preprocess(image);
    cached_image_path = image_path;
}
```

**バッチ処理**の実装は、複数の画像を連続して処理する際に、モデルの初期化コストをアモタイズし、全体的なスループットを向上させるための手法です。特に、GPUやマルチコアCPUを有効活用したい場合には、このアプローチが特に効果的です：

```cpp
// 複数画像の同時処理
std::vector<ClassificationResult> classify_batch(const std::vector<cv::Mat>& images) {
    std::vector<ClassificationResult> all_results;

    for (const auto& image : images) {
        auto results = classify_image(image);
        all_results.insert(all_results.end(), results.begin(), results.end());
    }

    return all_results;
}
```

**メモリプールの使用**については、頻繁なメモリ確保と解放が発生する状況でのパフォーマンス問題を解決するための手法です。カスタムアロケーターを設定することで、メモリの断片化を防ぎ、安定した高速なメモリアクセスを実現できます：

```cpp
// カスタムアロケーターの設定
ncnn::UnlockedPoolAllocator allocator;
net.opt.blob_allocator = &allocator;
net.opt.workspace_allocator = &allocator;
```