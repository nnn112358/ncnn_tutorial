# 5. C++ APIの基本使用法

## 5.1 必要なヘッダーファイル

ncnnを使用したC++プログラムを作成する際に、最低限必要なヘッダーファイルは以下の通りです：

```cpp
#include <ncnn/net.h>        // メインのncnn::Netクラス
#include <ncnn/mat.h>        // ncnn::Matテンソルクラス
#include <ncnn/layer.h>      // レイヤー基底クラス（カスタムレイヤー使用時）
#include <ncnn/option.h>     // 実行オプション設定
```

### 追加の便利なヘッダー

実際のアプリケーション開発では、ncnnのコアヘッダーに加えて、標準ライブラリやサードパーティライブラリのヘッダーも必要になります。標準入出力操作のための`<iostream>`、動的配列やコンテナ操作のための`<vector>`、そして画像処理機能が必要な場合は`<opencv2/opencv.hpp>`などが一般的に使用されます。

```cpp
#include <iostream>          // 標準入出力
#include <vector>            // STLコンテナ
#include <opencv2/opencv.hpp> // OpenCV（画像処理用、オプション）
```

### インクルードパスの設定

```cpp
// プリプロセッサマクロでのパス指定例
#ifdef USE_VULKAN
#include <ncnn/gpu.h>        // Vulkan GPU使用時
#endif
```

## 5.2 基本的なクラスとメソッド

### 5.2.1 ncnn::Net

`ncnn::Net`は、ニューラルネットワーク全体を管理するメインクラスです。

**主要なメソッド**:

```cpp
class Net {
public:
    // コンストラクタ/デストラクタ
    Net();
    ~Net();

    // モデル読み込み
    int load_param(const char* parampath);
    int load_param_bin(const char* binpath);
    int load_model(const char* parampath, const char* binpath);

    // メモリからの読み込み
    int load_param(const unsigned char* mem);
    int load_model(const unsigned char* param_mem, const unsigned char* bin_mem);

    // 推論実行
    ncnn::Extractor create_extractor() const;

    // 設定
    ncnn::Option opt;

    // リソース管理
    void clear();
};
```

**基本的な使用例**:

```cpp
#include <ncnn/net.h>

int main() {
    // ncnn::Netのインスタンス化
    ncnn::Net net;

    // 実行オプションの設定
    net.opt.use_vulkan_compute = false;  // CPU使用
    net.opt.num_threads = 4;             // スレッド数

    // モデルの読み込み
    int ret1 = net.load_param("model.param");
    int ret2 = net.load_model("model.param", "model.bin");

    if (ret1 != 0 || ret2 != 0) {
        fprintf(stderr, "Failed to load model\n");
        return -1;
    }

    // 推論処理...

    return 0;
}
```

### 5.2.2 ncnn::Mat

`ncnn::Mat`は、多次元テンソルデータを格納するクラスです。

**主要なメソッドと属性**:

```cpp
class Mat {
public:
    // コンストラクタ
    Mat();                                    // 空のマトリックス
    Mat(int w, size_t elemsize = 4u);        // 1次元
    Mat(int w, int h, size_t elemsize = 4u); // 2次元
    Mat(int w, int h, int c, size_t elemsize = 4u); // 3次元

    // データアクセス
    float* row(int y);                        // 行データへのポインタ
    const float* row(int y) const;
    template<typename T> T* channel(int c);   // チャンネルデータへのポインタ

    // 属性
    int w, h, c, dims;                       // 幅、高さ、チャンネル、次元数
    size_t cstep;                            // チャンネル間のステップサイズ

    // メモリ管理
    void create(int w, size_t elemsize = 4u);
    void create(int w, int h, size_t elemsize = 4u);
    void create(int w, int h, int c, size_t elemsize = 4u);
    void release();
    bool empty() const;

    // データ操作
    Mat clone() const;
    void fill(float v);
    Mat reshape(int w) const;
    Mat reshape(int w, int h) const;
};
```

**使用例**:

```cpp
// 3次元テンソルの作成 [3, 224, 224]
ncnn::Mat input(224, 224, 3);

// データの設定
input.fill(0.0f);  // 全要素を0で初期化

// 特定の要素へのアクセス
float* channel_0 = input.channel(0);
for (int h = 0; h < input.h; h++) {
    float* row_ptr = input.channel(0).row(h);
    for (int w = 0; w < input.w; w++) {
        row_ptr[w] = (float)(h * input.w + w);  // サンプルデータ
    }
}

// またはより簡潔な方法
float* ptr = input.channel(0);
for (int i = 0; i < input.w * input.h; i++) {
    ptr[i] = i * 0.01f;
}
```

### 5.2.3 ncnn::Extractor

`ncnn::Extractor`は、実際の推論処理を実行するクラスです。

**主要なメソッド**:

```cpp
class Extractor {
public:
    // 入力設定
    int input(const char* blob_name, const Mat& in);
    int input(int blob_index, const Mat& in);

    // 出力取得
    int extract(const char* blob_name, Mat& out);
    int extract(int blob_index, Mat& out);

    // 設定
    void set_light_mode(bool enable);
    void set_num_threads(int num_threads);
};
```

**使用例**:

```cpp
// Extractorの作成
ncnn::Extractor ex = net.create_extractor();

// 入力データの設定
ncnn::Mat input(224, 224, 3);
// ... inputにデータを設定 ...

// 推論実行
ex.input("input", input);

// 結果の取得
ncnn::Mat output;
ex.extract("output", output);

// 結果の処理
printf("Output shape: [%d, %d, %d]\n", output.c, output.h, output.w);
```

## 5.3 初回のHello Worldプログラム

実際に動作するシンプルなncnnプログラムを作成してみましょう。

### ソースコード (hello_ncnn.cpp)

```cpp
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <iostream>
#include <vector>

int main() {
    // ncnn初期化
    ncnn::Net net;

    // 実行オプションの設定
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 2;

    // モデルの読み込み（SqueezeNetを例とする）
    printf("Loading model...\n");
    int ret = net.load_model("squeezenet_v1.1.param", "squeezenet_v1.1.bin");
    if (ret != 0) {
        fprintf(stderr, "Failed to load model (ret=%d)\n", ret);
        return -1;
    }
    printf("Model loaded successfully!\n");

    // ダミー入力データの作成（224x224のRGB画像）
    ncnn::Mat input(224, 224, 3);

    // 簡単なグラデーションパターンでデータを初期化
    for (int c = 0; c < 3; c++) {
        float* channel_ptr = input.channel(c);
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                channel_ptr[h * 224 + w] = (float)(h + w + c * 100) / 1000.0f;
            }
        }
    }

    printf("Input prepared: [%d, %d, %d]\n", input.c, input.h, input.w);

    // 推論実行
    ncnn::Extractor ex = net.create_extractor();

    // 入力データの設定
    ex.input("data", input);

    // 推論実行と結果取得
    ncnn::Mat output;
    ex.extract("prob", output);

    printf("Inference completed!\n");
    printf("Output shape: [%d, %d, %d]\n", output.c, output.h, output.w);

    // 結果の表示（上位5つの値）
    std::vector<std::pair<float, int>> scores;
    const float* prob_data = output.channel(0);

    for (int i = 0; i < output.w; i++) {
        scores.push_back(std::make_pair(prob_data[i], i));
    }

    // スコアでソート（降順）
    std::sort(scores.begin(), scores.end(), std::greater<std::pair<float, int>>());

    // 上位5つの結果を表示
    printf("\nTop 5 predictions:\n");
    for (int i = 0; i < 5 && i < scores.size(); i++) {
        printf("  Class %d: %.6f\n", scores[i].second, scores[i].first);
    }

    return 0;
}
```

### エラーハンドリングの改良版

```cpp
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <iostream>
#include <stdexcept>

class NCNNInference {
private:
    ncnn::Net net;
    bool model_loaded;

public:
    NCNNInference() : model_loaded(false) {
        // デフォルト設定
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 4;
    }

    ~NCNNInference() {
        if (model_loaded) {
            net.clear();
        }
    }

    bool load_model(const std::string& param_path, const std::string& bin_path) {
        try {
            int ret1 = net.load_param(param_path.c_str());
            int ret2 = net.load_model(param_path.c_str(), bin_path.c_str());

            if (ret1 != 0) {
                std::cerr << "Failed to load param file: " << param_path << std::endl;
                return false;
            }

            if (ret2 != 0) {
                std::cerr << "Failed to load model file: " << bin_path << std::endl;
                return false;
            }

            model_loaded = true;
            std::cout << "Model loaded successfully!" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "Exception while loading model: " << e.what() << std::endl;
            return false;
        }
    }

    bool inference(const ncnn::Mat& input, ncnn::Mat& output,
                   const std::string& input_name = "data",
                   const std::string& output_name = "prob") {
        if (!model_loaded) {
            std::cerr << "Model not loaded!" << std::endl;
            return false;
        }

        try {
            ncnn::Extractor ex = net.create_extractor();

            int ret1 = ex.input(input_name.c_str(), input);
            if (ret1 != 0) {
                std::cerr << "Failed to set input: " << input_name << std::endl;
                return false;
            }

            int ret2 = ex.extract(output_name.c_str(), output);
            if (ret2 != 0) {
                std::cerr << "Failed to extract output: " << output_name << std::endl;
                return false;
            }

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Exception during inference: " << e.what() << std::endl;
            return false;
        }
    }
};

int main() {
    NCNNInference inference;

    // モデルの読み込み
    if (!inference.load_model("model.param", "model.bin")) {
        return -1;
    }

    // 入力データの準備
    ncnn::Mat input(224, 224, 3);
    input.fill(0.5f);  // 簡単な初期化

    // 推論実行
    ncnn::Mat output;
    if (!inference.inference(input, output)) {
        return -1;
    }

    std::cout << "Inference successful!" << std::endl;
    std::cout << "Output shape: [" << output.c << ", " << output.h << ", " << output.w << "]" << std::endl;

    return 0;
}
```

## 5.4 CMakeを使用したビルド設定

ncnnアプリケーションをビルドするためのCMakeListsファイルの作成方法を説明します。

### 基本的なCMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(hello_ncnn)

# C++標準の設定
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ncnnライブラリの検索
find_package(ncnn REQUIRED)

# 実行ファイルの定義
add_executable(hello_ncnn hello_ncnn.cpp)

# ncnnライブラリのリンク
target_link_libraries(hello_ncnn ncnn)

# コンパイルオプション
target_compile_options(hello_ncnn PRIVATE -O3 -Wall)
```

### 高度なCMakeLists.txt（OpenCV付き）

```cmake
cmake_minimum_required(VERSION 3.10)
project(ncnn_app)

# C++標準の設定
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ビルドタイプの設定
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ncnn ライブラリの検索
find_package(ncnn REQUIRED)

# OpenCV ライブラリの検索（オプション）
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_definitions(-DUSE_OPENCV)
endif()

# Vulkan サポートの確認
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_VERSION}")
    add_definitions(-DUSE_VULKAN)
endif()

# インクルードディレクトリの設定
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ソースファイルのグロビング
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")

# 実行ファイルの定義
add_executable(${PROJECT_NAME} ${SOURCES})

# ライブラリのリンク
target_link_libraries(${PROJECT_NAME} ncnn)

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

if(Vulkan_FOUND)
    target_link_libraries(${PROJECT_NAME} ${Vulkan_LIBRARIES})
endif()

# コンパイルオプション
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

# インストール設定
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
```

### ビルド手順

ncnnアプリケーションのビルドプロセスは、標準的なCMakeワークフローに従います。まず、プロジェクト用のディレクトリを作成し、必要なファイル（CMakeLists.txt、ソースコード、モデルファイル）を適切に配置します。次に、ビルド専用のディレクトリを作成してソースディレクトリとは分離し、CMakeによる設定とmakeによるコンパイルを実行します。この手順により、クリーンで管理しやすいビルド環境を構築できます。

```bash
# プロジェクトディレクトリの作成
mkdir my_ncnn_project
cd my_ncnn_project

# ソースファイルの配置
# - CMakeLists.txt
# - hello_ncnn.cpp
# - model.param
# - model.bin

# ビルドディレクトリの作成
mkdir build
cd build

# CMake設定
cmake ..

# または、特定のncnnインストールを指定
cmake -Dncnn_DIR=/path/to/ncnn/install/lib/cmake/ncnn ..

# ビルド実行
make -j$(nproc)

# 実行
./hello_ncnn
```

### pkgconfigを使用したビルド（代替方法）

CMakeを使用しない場合の代替手段として、pkgconfigを利用した直接的なコンパイル方法があります。この方法では、pkgconfigツールがncnnライブラリの適切なコンパイルフラグとリンクフラグを自動的に提供するため、シンプルなプロジェクトや学習目的での利用に適しています。まず、ncnnのpkgconfigファイルが正しく設定されているかを確認し、その後g++コンパイラを使用して直接的にビルドを実行します。

```bash
# pkgconfigファイルの確認
pkg-config --cflags ncnn
pkg-config --libs ncnn

# 直接的なg++コンパイル
g++ -o hello_ncnn hello_ncnn.cpp `pkg-config --cflags --libs ncnn` -std=c++11 -O3
```

### デバッグビルドの設定

開発段階やトラブルシューティング時には、デバッグ情報を含むビルドが重要になります。デバッグビルドでは、最適化を無効にしてデバッグシンボルを含めることで、デバッガーでの詳細な解析が可能になります。また、メモリ関連の問題を検出するために、アドレスサニタイザーなどの動的解析ツールを組み込むことも推奨されます。これらの設定により、開発効率の向上と品質の高いコードの作成が可能になります。

```cmake
# デバッグ情報の追加
set(CMAKE_BUILD_TYPE Debug)
target_compile_options(${PROJECT_NAME} PRIVATE -g -O0 -DDEBUG)

# アドレスサニタイザーの有効化（オプション）
target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
target_link_libraries(${PROJECT_NAME} -fsanitize=address)
```

これで、ncnnのC++ APIの基本的な使用方法を理解し、実際に動作するプログラムを作成できるようになりました。次章では、より実践的な画像分類アプリケーションの実装について詳しく学習します。