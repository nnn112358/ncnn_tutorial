# 7. 物体検出の実装

物体検出は、画像内の複数の物体を同時に認識し、その位置（バウンディングボックス）とクラスを特定するタスクです。本章では、ncnnを使用してYOLOシリーズを中心とした物体検出アプリケーションを実装する方法を詳しく解説します。

## 7.1 YOLOv5/YOLOv8モデルの使用

YOLOシリーズは、リアルタイム物体検出において最も人気の高いアーキテクチャの一つです。ncnnでは、YOLOv5とYOLOv8の両方が効率的にサポートされています。

### YOLOの基本概念

**YOLO（You Only Look Once）の特徴**について、このアーキテクチャは伝統的な物体検出手法とは異なり、画像を一度だけ解析することで全ての物体を同時に検出する革新的なアプローチです。このワンショット検出手法により、従来のスライディングウィンドウ方式よりも遥かに高速な推論速度を実現しています。また、物体の検出から分類までを統合したエンドツーエンドの学習アプローチを採用しているため、特にリアルタイム処理が求められるアプリケーションでの利用に適しています。

**出力テンソルの構造**:
```
YOLOv5/v8出力: [batch_size, num_proposals, 5 + num_classes]
- num_proposals: 提案領域の数（例：25200）
- 5: [center_x, center_y, width, height, objectness_score]
- num_classes: クラス数（COCOデータセットの場合80）
```

### YOLOモデルの準備

**PyTorchからの変換例**:

```python
# YOLOv5の場合
import torch

# 学習済みモデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# ダミー入力で推論グラフを作成
dummy_input = torch.randn(1, 3, 640, 640)

# ONNX形式でエクスポート
torch.onnx.export(
    model,
    dummy_input,
    "yolov5s.onnx",
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

```bash
# ONNXからncnnへの変換
onnx2ncnn yolov5s.onnx yolov5s.param yolov5s.bin
```

### YOLODetectorクラスの実装

```cpp
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

struct BoundingBox {
    float x1, y1, x2, y2;  // 左上と右下の座標
    float confidence;       // 信頼度
    int class_id;          // クラスID
    std::string class_name; // クラス名
};

class YOLODetector {
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    int input_width;
    int input_height;
    float confidence_threshold;
    float nms_threshold;
    bool model_loaded;

    // YOLOのアンカー情報（YOLOv5の場合）
    std::vector<std::vector<std::pair<int, int>>> anchors = {
        {{10, 13}, {16, 30}, {33, 23}},      // P3/8
        {{30, 61}, {62, 45}, {59, 119}},     // P4/16
        {{116, 90}, {156, 198}, {373, 326}}  // P5/32
    };

    std::vector<int> strides = {8, 16, 32};

public:
    YOLODetector(int width = 640, int height = 640,
                 float conf_thresh = 0.5f, float nms_thresh = 0.45f)
        : input_width(width), input_height(height),
          confidence_threshold(conf_thresh), nms_threshold(nms_thresh),
          model_loaded(false) {

        // ncnn設定
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
    }

    bool load_model(const std::string& param_path, const std::string& bin_path) {
        int ret1 = net.load_param(param_path.c_str());
        int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

        if (ret1 != 0 || ret2 != 0) {
            std::cerr << "Failed to load YOLO model" << std::endl;
            return false;
        }

        model_loaded = true;
        std::cout << "YOLO model loaded successfully" << std::endl;
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
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            class_names.push_back(line);
        }

        std::cout << "Loaded " << class_names.size() << " class names" << std::endl;
        return true;
    }

    ncnn::Mat preprocess_image(const cv::Mat& image, float& scale_factor) {
        // アスペクト比を保持したリサイズ
        int img_w = image.cols;
        int img_h = image.rows;

        float scale = std::min((float)input_width / img_w, (float)input_height / img_h);
        scale_factor = scale;

        int new_w = (int)(img_w * scale);
        int new_h = (int)(img_h * scale);

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));

        // パディングを追加してターゲットサイズにする
        cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_8UC3);
        int top = (input_height - new_h) / 2;
        int left = (input_width - new_w) / 2;

        resized.copyTo(padded(cv::Rect(left, top, new_w, new_h)));

        // RGB形式に変換
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

        // ncnn::Matに変換
        ncnn::Mat input = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, input_width, input_height);

        // 正規化 [0, 255] -> [0, 1]
        const float norm_vals[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
        input.substract_mean_normalize(0, norm_vals);

        return input;
    }

    std::vector<BoundingBox> detect(const cv::Mat& image) {
        if (!model_loaded) {
            throw std::runtime_error("Model not loaded");
        }

        // 前処理
        float scale_factor;
        ncnn::Mat input = preprocess_image(image, scale_factor);

        // 推論実行
        ncnn::Extractor ex = net.create_extractor();
        ex.input("images", input);

        ncnn::Mat output;
        ex.extract("output", output);

        // 後処理
        std::vector<BoundingBox> boxes = parse_yolo_output(output, scale_factor, image.cols, image.rows);

        // NMS適用
        boxes = apply_nms(boxes);

        return boxes;
    }

private:
    std::vector<BoundingBox> parse_yolo_output(const ncnn::Mat& output, float scale_factor,
                                               int orig_width, int orig_height) {
        std::vector<BoundingBox> boxes;

        int num_proposals = output.h;
        int num_values = output.w;  // 5 + num_classes

        const float* data = output.channel(0);

        for (int i = 0; i < num_proposals; i++) {
            const float* row = data + i * num_values;

            // バウンディングボックスの座標
            float center_x = row[0];
            float center_y = row[1];
            float width = row[2];
            float height = row[3];
            float objectness = row[4];

            if (objectness < confidence_threshold) {
                continue;
            }

            // クラススコアの中で最大値を見つける
            float max_class_score = 0.0f;
            int max_class_id = 0;

            for (int j = 5; j < num_values; j++) {
                if (row[j] > max_class_score) {
                    max_class_score = row[j];
                    max_class_id = j - 5;
                }
            }

            float final_score = objectness * max_class_score;
            if (final_score < confidence_threshold) {
                continue;
            }

            // 座標を元の画像サイズに変換
            float x1 = (center_x - width / 2.0f) / scale_factor;
            float y1 = (center_y - height / 2.0f) / scale_factor;
            float x2 = (center_x + width / 2.0f) / scale_factor;
            float y2 = (center_y + height / 2.0f) / scale_factor;

            // 画像境界内にクリップ
            x1 = std::max(0.0f, std::min((float)orig_width - 1, x1));
            y1 = std::max(0.0f, std::min((float)orig_height - 1, y1));
            x2 = std::max(0.0f, std::min((float)orig_width - 1, x2));
            y2 = std::max(0.0f, std::min((float)orig_height - 1, y2));

            BoundingBox box;
            box.x1 = x1;
            box.y1 = y1;
            box.x2 = x2;
            box.y2 = y2;
            box.confidence = final_score;
            box.class_id = max_class_id;

            if (max_class_id < class_names.size()) {
                box.class_name = class_names[max_class_id];
            } else {
                box.class_name = "Unknown";
            }

            boxes.push_back(box);
        }

        return boxes;
    }

    std::vector<BoundingBox> apply_nms(std::vector<BoundingBox>& boxes) {
        if (boxes.empty()) {
            return boxes;
        }

        // 信頼度で降順ソート
        std::sort(boxes.begin(), boxes.end(),
                  [](const BoundingBox& a, const BoundingBox& b) {
                      return a.confidence > b.confidence;
                  });

        std::vector<bool> suppressed(boxes.size(), false);
        std::vector<BoundingBox> result;

        for (size_t i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) {
                continue;
            }

            result.push_back(boxes[i]);

            // 他のボックスとのIoUを計算
            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j] || boxes[i].class_id != boxes[j].class_id) {
                    continue;
                }

                float iou = calculate_iou(boxes[i], boxes[j]);
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
        float intersection_x1 = std::max(box1.x1, box2.x1);
        float intersection_y1 = std::max(box1.y1, box2.y1);
        float intersection_x2 = std::min(box1.x2, box2.x2);
        float intersection_y2 = std::min(box1.y2, box2.y2);

        if (intersection_x2 <= intersection_x1 || intersection_y2 <= intersection_y1) {
            return 0.0f;
        }

        float intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1);

        float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

        float union_area = box1_area + box2_area - intersection_area;

        return intersection_area / union_area;
    }
};
```

## 7.2 バウンディングボックスの処理

検出されたバウンディングボックスの後処理と管理を行うクラスを実装します。

```cpp
class BoundingBoxProcessor {
public:
    // バウンディングボックスの有効性チェック
    static bool is_valid_box(const BoundingBox& box, int image_width, int image_height,
                            float min_size = 10.0f) {
        if (box.x1 >= box.x2 || box.y1 >= box.y2) {
            return false;
        }

        if (box.x1 < 0 || box.y1 < 0 || box.x2 >= image_width || box.y2 >= image_height) {
            return false;
        }

        float width = box.x2 - box.x1;
        float height = box.y2 - box.y1;

        return width >= min_size && height >= min_size;
    }

    // ボックスのサイズ計算
    static float calculate_box_area(const BoundingBox& box) {
        return (box.x2 - box.x1) * (box.y2 - box.y1);
    }

    // ボックスの中心座標計算
    static std::pair<float, float> get_box_center(const BoundingBox& box) {
        float center_x = (box.x1 + box.x2) / 2.0f;
        float center_y = (box.y1 + box.y2) / 2.0f;
        return {center_x, center_y};
    }

    // 信頼度によるフィルタリング
    static std::vector<BoundingBox> filter_by_confidence(const std::vector<BoundingBox>& boxes,
                                                        float min_confidence) {
        std::vector<BoundingBox> filtered;
        for (const auto& box : boxes) {
            if (box.confidence >= min_confidence) {
                filtered.push_back(box);
            }
        }
        return filtered;
    }

    // 特定のクラスでフィルタリング
    static std::vector<BoundingBox> filter_by_classes(const std::vector<BoundingBox>& boxes,
                                                     const std::vector<int>& target_classes) {
        std::vector<BoundingBox> filtered;
        for (const auto& box : boxes) {
            if (std::find(target_classes.begin(), target_classes.end(), box.class_id) != target_classes.end()) {
                filtered.push_back(box);
            }
        }
        return filtered;
    }

    // ボックスのサイズでフィルタリング
    static std::vector<BoundingBox> filter_by_size(const std::vector<BoundingBox>& boxes,
                                                  float min_area, float max_area = -1.0f) {
        std::vector<BoundingBox> filtered;
        for (const auto& box : boxes) {
            float area = calculate_box_area(box);
            if (area >= min_area && (max_area < 0 || area <= max_area)) {
                filtered.push_back(box);
            }
        }
        return filtered;
    }
};
```

## 7.3 Non-Maximum Suppression（NMS）

より高度なNMS実装とカスタマイズオプションを提供します。

```cpp
class AdvancedNMS {
public:
    struct NMSConfig {
        float iou_threshold = 0.45f;
        float score_threshold = 0.5f;
        int max_detections = 100;
        bool class_agnostic = false;  // クラスに関係なくNMSを適用
        bool soft_nms = false;        // Soft-NMSを使用
        float soft_nms_sigma = 0.5f;  // Soft-NMSのシグマパラメータ
    };

    static std::vector<BoundingBox> apply_nms(std::vector<BoundingBox> boxes,
                                            const NMSConfig& config = NMSConfig()) {
        if (boxes.empty()) {
            return boxes;
        }

        // スコアでフィルタリング
        boxes.erase(
            std::remove_if(boxes.begin(), boxes.end(),
                          [&config](const BoundingBox& box) {
                              return box.confidence < config.score_threshold;
                          }),
            boxes.end()
        );

        if (config.soft_nms) {
            return apply_soft_nms(boxes, config);
        } else {
            return apply_hard_nms(boxes, config);
        }
    }

private:
    static std::vector<BoundingBox> apply_hard_nms(std::vector<BoundingBox> boxes,
                                                  const NMSConfig& config) {
        // 信頼度で降順ソート
        std::sort(boxes.begin(), boxes.end(),
                  [](const BoundingBox& a, const BoundingBox& b) {
                      return a.confidence > b.confidence;
                  });

        std::vector<bool> suppressed(boxes.size(), false);
        std::vector<BoundingBox> result;

        for (size_t i = 0; i < boxes.size() && result.size() < config.max_detections; i++) {
            if (suppressed[i]) {
                continue;
            }

            result.push_back(boxes[i]);

            for (size_t j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }

                // クラス固有のNMSの場合、同じクラスのみ比較
                if (!config.class_agnostic && boxes[i].class_id != boxes[j].class_id) {
                    continue;
                }

                float iou = calculate_iou(boxes[i], boxes[j]);
                if (iou > config.iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    static std::vector<BoundingBox> apply_soft_nms(std::vector<BoundingBox> boxes,
                                                  const NMSConfig& config) {
        std::vector<BoundingBox> result;

        while (!boxes.empty() && result.size() < config.max_detections) {
            // 最大スコアのボックスを見つける
            auto max_it = std::max_element(boxes.begin(), boxes.end(),
                                         [](const BoundingBox& a, const BoundingBox& b) {
                                             return a.confidence < b.confidence;
                                         });

            if (max_it->confidence < config.score_threshold) {
                break;
            }

            BoundingBox selected = *max_it;
            result.push_back(selected);
            boxes.erase(max_it);

            // 残りのボックスのスコアを調整
            for (auto& box : boxes) {
                if (!config.class_agnostic && box.class_id != selected.class_id) {
                    continue;
                }

                float iou = calculate_iou(selected, box);
                if (iou > 0) {
                    // Soft-NMSによるスコア調整
                    float weight = std::exp(-(iou * iou) / config.soft_nms_sigma);
                    box.confidence *= weight;
                }
            }
        }

        return result;
    }

    static float calculate_iou(const BoundingBox& box1, const BoundingBox& box2) {
        float intersection_x1 = std::max(box1.x1, box2.x1);
        float intersection_y1 = std::max(box1.y1, box2.y1);
        float intersection_x2 = std::min(box1.x2, box2.x2);
        float intersection_y2 = std::min(box1.y2, box2.y2);

        if (intersection_x2 <= intersection_x1 || intersection_y2 <= intersection_y1) {
            return 0.0f;
        }

        float intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1);
        float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
        float union_area = box1_area + box2_area - intersection_area;

        return intersection_area / union_area;
    }
};
```

## 7.4 検出結果の可視化

検出結果を視覚的に表示するための包括的な可視化クラスを実装します。

```cpp
class DetectionVisualizer {
public:
    struct VisualizationConfig {
        cv::Scalar box_color = cv::Scalar(0, 255, 0);      // 緑色
        cv::Scalar text_color = cv::Scalar(255, 255, 255);  // 白色
        cv::Scalar bg_color = cv::Scalar(0, 0, 0);         // 黒色背景
        int box_thickness = 2;
        int text_thickness = 1;
        double font_scale = 0.6;
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        bool show_confidence = true;
        bool show_class_name = true;
        bool draw_filled_background = true;
    };

    static void draw_detections(cv::Mat& image, const std::vector<BoundingBox>& boxes,
                              const VisualizationConfig& config = VisualizationConfig()) {
        for (const auto& box : boxes) {
            draw_single_detection(image, box, config);
        }
    }

    static void draw_single_detection(cv::Mat& image, const BoundingBox& box,
                                    const VisualizationConfig& config = VisualizationConfig()) {
        // バウンディングボックスの描画
        cv::Point pt1(static_cast<int>(box.x1), static_cast<int>(box.y1));
        cv::Point pt2(static_cast<int>(box.x2), static_cast<int>(box.y2));

        cv::rectangle(image, pt1, pt2, config.box_color, config.box_thickness);

        // ラベルテキストの準備
        std::string label = create_label(box, config);

        if (!label.empty()) {
            // テキストサイズの計算
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, config.font_face,
                                               config.font_scale, config.text_thickness, &baseline);

            // テキスト位置の計算
            cv::Point text_origin(pt1.x, pt1.y - baseline - 5);

            // テキストが画像外に出る場合の調整
            if (text_origin.y < text_size.height) {
                text_origin.y = pt1.y + text_size.height + 5;
            }

            // 背景矩形の描画
            if (config.draw_filled_background) {
                cv::Point bg_pt1(text_origin.x, text_origin.y - text_size.height);
                cv::Point bg_pt2(text_origin.x + text_size.width, text_origin.y + baseline);
                cv::rectangle(image, bg_pt1, bg_pt2, config.bg_color, cv::FILLED);
            }

            // テキストの描画
            cv::putText(image, label, text_origin, config.font_face,
                       config.font_scale, config.text_color, config.text_thickness);
        }
    }

    static void draw_detection_statistics(cv::Mat& image, const std::vector<BoundingBox>& boxes) {
        // 検出統計の計算
        std::map<std::string, int> class_counts;
        for (const auto& box : boxes) {
            class_counts[box.class_name]++;
        }

        // 統計情報の描画
        int y_offset = 30;
        std::string total_text = "Total detections: " + std::to_string(boxes.size());
        cv::putText(image, total_text, cv::Point(10, y_offset),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        y_offset += 30;
        for (const auto& pair : class_counts) {
            std::string class_text = pair.first + ": " + std::to_string(pair.second);
            cv::putText(image, class_text, cv::Point(10, y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            y_offset += 25;
        }
    }

    static cv::Mat create_detection_overlay(const cv::Mat& original,
                                          const std::vector<BoundingBox>& boxes,
                                          float alpha = 0.7f) {
        cv::Mat overlay = original.clone();
        cv::Mat result;

        draw_detections(overlay, boxes);
        cv::addWeighted(original, 1.0f - alpha, overlay, alpha, 0, result);

        return result;
    }

    // カラーパレットの生成（クラスごとに異なる色）
    static cv::Scalar get_class_color(int class_id) {
        static std::vector<cv::Scalar> colors = {
            cv::Scalar(255, 0, 0),    // 赤
            cv::Scalar(0, 255, 0),    // 緑
            cv::Scalar(0, 0, 255),    // 青
            cv::Scalar(255, 255, 0),  // シアン
            cv::Scalar(255, 0, 255),  // マゼンタ
            cv::Scalar(0, 255, 255),  // 黄色
            cv::Scalar(128, 0, 128),  // 紫
            cv::Scalar(255, 165, 0),  // オレンジ
            cv::Scalar(0, 128, 128),  // ティール
            cv::Scalar(128, 128, 0),  // オリーブ
        };

        return colors[class_id % colors.size()];
    }

private:
    static std::string create_label(const BoundingBox& box, const VisualizationConfig& config) {
        std::stringstream ss;

        if (config.show_class_name && !box.class_name.empty()) {
            ss << box.class_name;
        }

        if (config.show_confidence) {
            if (config.show_class_name) {
                ss << " ";
            }
            ss << std::fixed << std::setprecision(2) << box.confidence;
        }

        return ss.str();
    }
};

// 使用例
int main() {
    // YOLODetectorの初期化
    YOLODetector detector(640, 640, 0.5f, 0.45f);

    if (!detector.load_model("yolov5s.param", "yolov5s.bin")) {
        return -1;
    }

    detector.load_class_names("coco_classes.txt");

    // 画像の読み込み
    cv::Mat image = cv::imread("test_image.jpg");

    // 物体検出
    auto boxes = detector.detect(image);

    // 高度なNMSの適用
    AdvancedNMS::NMSConfig nms_config;
    nms_config.iou_threshold = 0.45f;
    nms_config.soft_nms = true;

    boxes = AdvancedNMS::apply_nms(boxes, nms_config);

    // 結果の可視化
    DetectionVisualizer::draw_detections(image, boxes);
    DetectionVisualizer::draw_detection_statistics(image, boxes);

    // 結果の保存
    cv::imwrite("detection_result.jpg", image);

    return 0;
}
```

この実装により、高精度で高速な物体検出アプリケーションが完成しました。YOLOv5/v8モデルを使用して、リアルタイムでの物体検出が可能になります。次章では、さらなるパフォーマンス最適化について学習します。