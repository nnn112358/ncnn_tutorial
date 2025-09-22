# 12. 参考資料

本章では、ncnn開発をさらに深めるための公式ドキュメント、サンプルコード集、コミュニティリソース、関連ツールなどの参考情報をまとめます。

## 12.1 公式ドキュメント

### ncnn公式リポジトリ

**GitHub**: https://github.com/Tencent/ncnn

ncnnの最新ソースコード、ドキュメント、サンプルコードが公開されています。

**主要なドキュメント**について、プロジェクトの理解を深めるために重要なファイルとディレクトリが整理されています。`README.md`ファイルにはプロジェクトの概要と基本的な使用方法が記載されており、`docs/`ディレクトリには詳細な技術ドキュメントが、`examples/`には実用的なサンプルコード、`benchmark/`にはパフォーマンステスト用のコードがそれぞれ格納されています。

### API リファレンス

**ncnn::Net クラス**:
```cpp
// モデル読み込み
int load_param(const char* parampath);
int load_model(const char* parampath, const char* binpath);

// 推論実行
ncnn::Extractor create_extractor() const;

// 設定オプション
ncnn::Option opt;
```

**ncnn::Mat クラス**:
```cpp
// コンストラクタ
Mat();
Mat(int w, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
Mat(int w, int h, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
Mat(int w, int h, int c, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

// データアクセス
float* row(int y);
template<typename T> T* channel(int c);

// 形状操作
Mat clone(ncnn::Allocator* allocator = 0) const;
Mat reshape(int w, ncnn::Allocator* allocator = 0) const;
```

**ncnn::Extractor クラス**:
```cpp
// 入力設定
int input(const char* blob_name, const Mat& in);
int input(int blob_index, const Mat& in);

// 出力取得
int extract(const char* blob_name, Mat& out);
int extract(int blob_index, Mat& out);

// 設定
void set_light_mode(bool enable);
void set_num_threads(int num_threads);
```

### ビルド設定オプション

**主要なCMakeオプション**:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `NCNN_VULKAN` | ON | Vulkan GPU サポート |
| `NCNN_OPENMP` | ON | OpenMP 並列処理 |
| `NCNN_THREADS` | ON | pthread スレッド |
| `NCNN_BENCHMARK` | OFF | ベンチマークツール |
| `NCNN_BUILD_EXAMPLES` | ON | サンプルコード |
| `NCNN_BUILD_TOOLS` | ON | 変換ツール |
| `NCNN_PYTHON` | OFF | Python バインディング |
| `NCNN_INT8` | ON | 8bit 量子化サポート |
| `NCNN_BF16` | ON | BFloat16 サポート |
| `NCNN_ARM82` | OFF | ARMv8.2 命令セット |

**プラットフォーム固有の設定**:
```cmake
# Android向けビルド
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM=android-21 \
      -DNCNN_VULKAN=ON \
      ..

# iOS向けビルド
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake \
      -DPLATFORM=OS64 \
      -DDEPLOYMENT_TARGET="9.0" \
      ..

# Windows向けビルド（Visual Studio）
cmake -G "Visual Studio 16 2019" -A x64 \
      -DNCNN_VULKAN=ON \
      ..
```

## 12.2 サンプルコード集

### 基本的な推論サンプル

**画像分類 (squeezenet.cpp)**:
```cpp
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

int main() {
    ncnn::Net squeezenet;
    squeezenet.opt.use_vulkan_compute = true;

    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    cv::Mat img = cv::imread("image.jpg");
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(227, 227));

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        resized.data, ncnn::Mat::PIXEL_BGR, 227, 227, 227, 227);

    const float mean_vals[3] = {104.f, 177.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    return 0;
}
```

**物体検出 (yolov5.cpp)**:
```cpp
#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects) {
    ncnn::Net yolov5;
    yolov5.opt.use_vulkan_compute = true;

    yolov5.load_param("yolov5s.param");
    yolov5.load_model("yolov5s.bin");

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    const int target_size = 640;

    // letterbox resize
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                          wpad / 2, wpad - wpad / 2,
                          ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov5.create_extractor();
    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("output", out);

    // 後処理（NMS等）は省略

    return 0;
}
```

### プラットフォーム固有サンプル

**Android JNI インターフェース**:
```cpp
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <jni.h>

static ncnn::Net g_net;
static ncnn::Mutex lock;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {}

JNIEXPORT jboolean JNICALL
Java_com_example_ncnndemo_MainActivity_loadModel(JNIEnv* env, jobject thiz,
                                                 jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    bool use_gpu = ncnn::get_gpu_count() > 0;
    g_net.opt.use_vulkan_compute = use_gpu;
    g_net.opt.use_fp16_packed = true;
    g_net.opt.use_fp16_storage = true;

    int ret = g_net.load_param(mgr, "model.param");
    if (ret != 0) return JNI_FALSE;

    ret = g_net.load_model(mgr, "model.bin");
    if (ret != 0) return JNI_FALSE;

    return JNI_TRUE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_example_ncnndemo_MainActivity_detect(JNIEnv* env, jobject thiz,
                                              jobject bitmap) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return NULL;
    }

    void* pixels;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    // 推論処理
    ncnn::Mat in = ncnn::Mat::from_pixels((const unsigned char*)pixels,
                                         ncnn::Mat::PIXEL_RGBA2RGB,
                                         info.width, info.height);

    // 前処理、推論、後処理...

    AndroidBitmap_unlockPixels(env, bitmap);

    // 結果をJavaオブジェクトに変換
    return NULL;
}

}
```

**iOS Objective-C++ インターフェース**:
```objc
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <ncnn/net.h>

@interface NCNNDetector : NSObject

@property (nonatomic, assign) ncnn::Net* net;

- (BOOL)loadModel:(NSString*)paramPath binPath:(NSString*)binPath;
- (NSArray*)detectObjects:(UIImage*)image;

@end

@implementation NCNNDetector

- (instancetype)init {
    self = [super init];
    if (self) {
        _net = new ncnn::Net();
        _net->opt.use_vulkan_compute = false; // iOS metal backend
    }
    return self;
}

- (void)dealloc {
    delete _net;
}

- (BOOL)loadModel:(NSString*)paramPath binPath:(NSString*)binPath {
    int ret = _net->load_param([paramPath UTF8String]);
    if (ret != 0) return NO;

    ret = _net->load_model([paramPath UTF8String], [binPath UTF8String]);
    return ret == 0;
}

- (NSArray*)detectObjects:(UIImage*)image {
    // UIImage -> cv::Mat 変換
    CGImageRef imageRef = image.CGImage;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();

    cv::Mat mat((int)image.size.height, (int)image.size.width, CV_8UC4);
    CGContextRef contextRef = CGBitmapContextCreate(
        mat.data, mat.cols, mat.rows, 8, mat.step[0],
        colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);

    CGContextDrawImage(contextRef,
                      CGRectMake(0, 0, mat.cols, mat.rows), imageRef);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);

    // BGR変換
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);

    // ncnn推論
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    ncnn::Extractor ex = _net->create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    // 結果をNSArrayに変換
    NSMutableArray* results = [NSMutableArray array];
    // ... 結果処理 ...

    return results;
}

@end
```

## 12.3 コミュニティリソース

### フォーラムとディスカッション

**GitHub Issues** (https://github.com/Tencent/ncnn/issues) では、バグレポート、機能要求、技術的な質問などの具体的な問題が議論されています。一方、**GitHub Discussions** (https://github.com/Tencent/ncnn/discussions) は、一般的な質問やベストプラクティスの共有、コミュニティメンバー間でのより広範なディスカッションのためのプラットフォームとして活用されています。

### よく参照される外部リソース

**学術論文とテクニカルペーパー**:

1. **ncnn関連の論文**:
   - "ncnn: A High-Performance Neural Network Inference Framework Optimized for the Mobile Platform"
   - "Efficient Inference of Deep Neural Networks on Mobile Devices"

2. **最適化技術に関する論文**:
   - "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
   - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
   - "Mixed Precision Training" (ICLR 2018)

**関連ブログとチュートリアル**:
- Tencent AI Lab ブログ
- ONNX公式ドキュメント
- Vulkan開発者ガイド
- ARM開発者リソース

### コミュニティプロジェクト

**ncnn-based プロジェクト**:

1. **RealESRGAN-ncnn-vulkan**:
   - 超解像度アプリケーション
   - GitHub: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan

2. **waifu2x-ncnn-vulkan**:
   - アニメ画像の超解像度・ノイズ除去
   - GitHub: https://github.com/nihui/waifu2x-ncnn-vulkan

3. **SCRFD-ncnn**:
   - 顔検出アプリケーション
   - GitHub: https://github.com/nihui/SCRFD-ncnn

**言語バインディング**として、様々なプログラミング言語からncnnを利用できるライブラリが開発されています。公式のPythonバインディングである**ncnn-python**をはじめ、Rustエコシステム向けの**ncnn-rs**、ウェブ開発向けの**ncnn.js** (JavaScript/WebAssembly) などが提供されており、開発者の好みや用途に応じて選択することができます。

## 12.4 関連ツール

### モデル変換ツール

**公式ツール** (ncnn/tools/):
```bash
# ONNX変換
onnx2ncnn model.onnx model.param model.bin

# PyTorch変換（pnnx経由）
pnnx model.pt inputshape=[1,3,224,224]
pnnx2ncnn model.pnnx.param model.pnnx.bin model.param model.bin

# Caffe変換
caffe2ncnn deploy.prototxt model.caffemodel model.param model.bin

# 量子化
ncnn2int8 model.param model.bin quantized.param quantized.bin images.txt

# 最適化
ncnnoptimize model.param model.bin optimized.param optimized.bin 65536
```

**サードパーティツール**:

1. **PINTO model zoo**:
   - 変換済みモデルの大規模コレクション
   - GitHub: https://github.com/PINTO0309/PINTO_model_zoo

2. **onnx-simplifier**:
   - ONNX モデルの最適化
   - `pip install onnx-simplifier`

3. **netron**:
   - ニューラルネットワークモデルの可視化
   - https://netron.app/

### 開発支援ツール

**パフォーマンス測定**:
```bash
# ベンチマークツール
./benchncnn 10 4 0 -1 1

# カスタムベンチマーク
./benchncnn 10 1 0 0 0 model.param model.bin

# GPU ベンチマーク
./benchncnn 10 1 0 0 1 model.param model.bin
```

**デバッグツール**:
```bash
# モデル情報表示
./ncnn2table model.param model.bin

# レイヤー詳細情報
./ncnn_debug model.param model.bin
```

**プロファイリングツール**については、様々なプラットフォームに対応した専門ツールが利用可能です。GNU環境での標準的な**gprof**、Linux系システムでの包括的なパフォーマンス解析を行う**perf**、Intel プロセッサに特化した高度な解析機能を提供する**Intel VTune**、そしてARM アーキテクチャ向けに最適化された**ARM Performance Studio**などがあり、それぞれの開発環境や対象プラットフォームに応じて選択することができます。

### 統合開発環境とプラグイン

**Visual Studio Code**:
- C/C++ Extension Pack
- CMake Tools
- GitHub Copilot (コード補完)

**CLion**:
- JetBrains製 C++ IDE
- CMake統合
- デバッガー統合

**Xcode** (macOS/iOS開発):
- Metal Performance Shaders
- Instruments (プロファイリング)

### CI/CD 設定例

**GitHub Actions**:
```yaml
name: Build ncnn

on: [push, pull_request]

jobs:
  ubuntu:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
      with:
        submodules: true

    - name: Install dependencies
      run: |
        sudo apt update
        sudo apt install build-essential cmake libprotobuf-dev protobuf-compiler

    - name: Configure
      run: |
        mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DNCNN_VULKAN=OFF \
              -DNCNN_BUILD_EXAMPLES=ON \
              ..

    - name: Build
      run: |
        cd build
        make -j$(nproc)

    - name: Test
      run: |
        cd build
        ctest --output-on-failure

  android:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
      with:
        submodules: true

    - name: Setup Android NDK
      uses: nttld/setup-ndk@v1
      with:
        ndk-version: r25c

    - name: Build for Android
      run: |
        mkdir build-android && cd build-android
        cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
              -DANDROID_ABI="arm64-v8a" \
              -DANDROID_PLATFORM=android-21 \
              -DNCNN_VULKAN=ON \
              ..
        make -j$(nproc)
```

これらの参考資料を活用することで、ncnnの学習を継続し、コミュニティと連携しながら高品質なアプリケーションを開発することができます。