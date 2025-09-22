# 付録

## 付録A. APIリファレンス

### A.1 ncnn::Net クラス

ncnnの中核となるネットワーククラスの完全なAPIリファレンスです。

#### A.1.1 コンストラクタとデストラクタ

```cpp
class Net {
public:
    // デフォルトコンストラクタ
    Net();

    // デストラクタ
    ~Net();

    // コピーコンストラクタ（削除済み）
    Net(const Net&) = delete;

    // 代入演算子（削除済み）
    Net& operator=(const Net&) = delete;
};
```

#### A.1.2 モデル読み込みメソッド

```cpp
// パラメータファイルの読み込み
int load_param(const char* parampath);
int load_param(const unsigned char* mem, int memsize);
int load_param_bin(const char* binpath);

// モデルファイルの読み込み
int load_model(const char* binpath);
int load_model(const unsigned char* mem, int memsize);

// 統合読み込み
int load_model(const char* parampath, const char* binpath);

// Android Asset からの読み込み
#ifdef __ANDROID_API__
int load_param(AAssetManager* mgr, const char* assetpath);
int load_model(AAssetManager* mgr, const char* assetpath);
int load_model(AAssetManager* mgr, const char* paramassetpath, const char* binassetpath);
#endif
```

**戻り値**:
- `0`: 成功
- `-1`: ファイルオープンエラー
- `-2`: フォーマットエラー
- `-100`: メモリ不足

#### A.1.3 推論実行メソッド

```cpp
// Extractorの作成
ncnn::Extractor create_extractor() const;

// カスタムレイヤーの登録
typedef ncnn::Layer* (*layer_creator_func)(void*);
int register_custom_layer(const char* type, layer_creator_func creator, void* userdata = 0);

// リソース解放
void clear();
```

#### A.1.4 設定オプション

```cpp
class Net {
public:
    ncnn::Option opt;  // 実行オプション
};

class Option {
public:
    // スレッド設定
    int num_threads;                    // スレッド数（デフォルト: 1）

    // メモリ設定
    ncnn::Allocator* blob_allocator;    // ブロブ用アロケーター
    ncnn::Allocator* workspace_allocator; // ワークスペース用アロケーター
    bool use_memory_pool;               // メモリプール使用（デフォルト: true）

    // 最適化設定
    bool use_winograd_convolution;      // Winograd畳み込み（デフォルト: true）
    bool use_sgemm_convolution;         // SGEMM畳み込み（デフォルト: true）
    bool use_int8_inference;            // INT8推論（デフォルト: false）
    bool use_packing_layout;            // パッキングレイアウト（デフォルト: true）
    bool use_fp16_packed;               // FP16パッキング（デフォルト: false）
    bool use_fp16_storage;              // FP16ストレージ（デフォルト: false）
    bool use_fp16_arithmetic;           // FP16演算（デフォルト: false）
    bool use_int8_storage;              // INT8ストレージ（デフォルト: false）
    bool use_int8_arithmetic;           // INT8演算（デフォルト: false）
    bool use_bf16_storage;              // BF16ストレージ（デフォルト: false）

    // Vulkan設定
    bool use_vulkan_compute;            // Vulkan使用（デフォルト: false）
    int vulkan_device;                  // Vulkanデバイス番号（デフォルト: 0）

    // その他
    bool use_local_memory_pool;         // ローカルメモリプール（デフォルト: false）
    bool use_image_storage;             // イメージストレージ（デフォルト: false）
    bool use_tensor_storage;            // テンソルストレージ（デフォルト: false）
};
```

### A.2 ncnn::Mat クラス

多次元テンソルを表現するクラスの詳細仕様です。

#### A.2.1 コンストラクタ

```cpp
class Mat {
public:
    // 空のマトリックス
    Mat();

    // 1次元マトリックス
    Mat(int w, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

    // 2次元マトリックス
    Mat(int w, int h, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

    // 3次元マトリックス
    Mat(int w, int h, int c, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

    // 4次元マトリックス
    Mat(int w, int h, int d, int c, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

    // データポインタから作成
    Mat(int w, void* data, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
    Mat(int w, int h, void* data, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
    Mat(int w, int h, int c, void* data, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

    // コピーコンストラクタ
    Mat(const Mat& m);

    // デストラクタ
    ~Mat();
};
```

#### A.2.2 静的ファクトリーメソッド

```cpp
// 画像データから作成
static Mat from_pixels(const unsigned char* pixels, int type, int w, int h, ncnn::Allocator* allocator = 0);
static Mat from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, ncnn::Allocator* allocator = 0);
static Mat from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, ncnn::Allocator* allocator = 0);
static Mat from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, ncnn::Allocator* allocator = 0);

// ピクセルタイプ定数
enum PixelType {
    PIXEL_RGB       = 1,
    PIXEL_BGR       = 2,
    PIXEL_GRAY      = 3,
    PIXEL_RGBA      = 4,
    PIXEL_BGRA      = 5,
    PIXEL_RGB2BGR   = 6,
    PIXEL_RGB2GRAY  = 7,
    PIXEL_BGR2GRAY  = 8,
    PIXEL_GRAY2RGB  = 9,
    PIXEL_GRAY2BGR  = 10,
    PIXEL_RGBA2RGB  = 11,
    PIXEL_RGBA2BGR  = 12,
    PIXEL_RGBA2GRAY = 13,
    PIXEL_BGRA2RGB  = 14,
    PIXEL_BGRA2BGR  = 15,
    PIXEL_BGRA2GRAY = 16
};
```

#### A.2.3 データアクセスメソッド

```cpp
// 要素アクセス
template<typename T> T* row(int y);
template<typename T> const T* row(int y) const;
template<typename T> T* channel(int c);
template<typename T> const T* channel(int c) const;

// 便利なアクセサ
float* row(int y) { return row<float>(y); }
const float* row(int y) const { return row<float>(y); }
float* channel(int c) { return channel<float>(c); }
const float* channel(int c) const { return channel<float>(c); }

// 演算子オーバーロード
float& operator[](size_t i);
const float& operator[](size_t i) const;
```

#### A.2.4 形状操作メソッド

```cpp
// メモリ確保
void create(int w, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
void create(int w, int h, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
void create(int w, int h, int c, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);
void create(int w, int h, int d, int c, size_t elemsize = 4u, ncnn::Allocator* allocator = 0);

// 形状変更
Mat reshape(int w, ncnn::Allocator* allocator = 0) const;
Mat reshape(int w, int h, ncnn::Allocator* allocator = 0) const;
Mat reshape(int w, int h, int c, ncnn::Allocator* allocator = 0) const;

// クローン
Mat clone(ncnn::Allocator* allocator = 0) const;

// チャンネル範囲
Mat channel_range(int c, int channels) const;
Mat row_range(int y, int rows) const;
Mat range(int x, int n) const;

// データ操作
void fill(float v);
void fill(int v);

// 正規化
void substract_mean_normalize(const float* mean_vals, const float* norm_vals);

// 画像変換
void to_pixels(unsigned char* pixels, int type) const;
void to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const;
```

#### A.2.5 属性とプロパティ

```cpp
class Mat {
public:
    // 形状情報
    int dims;           // 次元数 (1, 2, 3, 4)
    int w;              // 幅
    int h;              // 高さ
    int d;              // 奥行き
    int c;              // チャンネル数

    // メモリ情報
    size_t elemsize;    // 要素サイズ (1, 2, 4 bytes)
    int elempack;       // 要素パック数

    // データポインタ
    void* data;         // データの先頭ポインタ

    // ステップサイズ
    size_t cstep;       // チャンネル間ステップ

    // 状態確認
    bool empty() const;         // 空かどうか
    size_t total() const;       // 総要素数

    // メモリ解放
    void release();

    // アロケーター
    ncnn::Allocator* allocator; // 使用中のアロケーター
};
```

### A.3 ncnn::Extractor クラス

推論実行を管理するクラスの詳細仕様です。

#### A.3.1 基本メソッド

```cpp
class Extractor {
public:
    // 入力設定
    int input(const char* blob_name, const Mat& in);
    int input(int blob_index, const Mat& in);

    // 出力取得
    int extract(const char* blob_name, Mat& out);
    int extract(int blob_index, Mat& out);

    // 複数出力取得
    int extract(const char* blob_name, Mat& out, int type);
    int extract(int blob_index, Mat& out, int type);
};
```

#### A.3.2 設定メソッド

```cpp
// 軽量モード設定
void set_light_mode(bool enable);

// スレッド数設定
void set_num_threads(int num_threads);

// Vulkan設定
void set_vulkan_compute(bool enable);

// ブロブアロケーター設定
void set_blob_allocator(ncnn::Allocator* allocator);

// ワークスペースアロケーター設定
void set_workspace_allocator(ncnn::Allocator* allocator);

// ブロブメモリバジェット設定（MB単位）
void set_blob_vkallocator(ncnn::VkAllocator* allocator);
void set_workspace_vkallocator(ncnn::VkAllocator* allocator);
void set_staging_vkallocator(ncnn::VkAllocator* allocator);
```

## 付録B. サポートされるオペレーター一覧

### B.1 基本演算レイヤー

| レイヤータイプ | ncnn名 | 説明 | 主要パラメータ |
|--------------|-------|------|---------------|
| **畳み込み** | Convolution | 2D畳み込み演算 | num_output, kernel_size, stride, pad |
| **畳み込み1D** | Convolution1D | 1D畳み込み演算 | num_output, kernel_size, stride, pad |
| **Depthwise畳み込み** | ConvolutionDepthWise | チャンネル単位畳み込み | num_output, kernel_size, stride, pad |
| **Deconvolution** | Deconvolution | 転置畳み込み | num_output, kernel_size, stride, pad |
| **全結合** | InnerProduct | 線形変換 | num_output, bias_term |

### B.2 活性化関数レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **ReLU** | ReLU | ReLU活性化 | negative_slope |
| **ReLU6** | ReLU6 | ReLU6活性化 | - |
| **Leaky ReLU** | ReLU | Leaky ReLU | negative_slope |
| **PReLU** | PReLU | Parametric ReLU | num_slopes |
| **ELU** | ELU | ELU活性化 | alpha |
| **SELU** | SELU | SELU活性化 | alpha, beta |
| **Sigmoid** | Sigmoid | シグモイド | - |
| **Tanh** | TanH | ハイパボリックタンジェント | - |
| **Swish** | Swish | Swish活性化 | - |
| **Mish** | Mish | Mish活性化 | - |
| **HardSwish** | HardSwish | Hard Swish | alpha, beta |
| **GELU** | GELU | GELU活性化 | - |

### B.3 プーリングレイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **Max Pooling** | Pooling | 最大プーリング | pooling_type=0, kernel_size, stride, pad |
| **Average Pooling** | Pooling | 平均プーリング | pooling_type=1, kernel_size, stride, pad |
| **Global Average Pooling** | Pooling | グローバル平均プーリング | global_pooling=1 |
| **Adaptive Pooling** | AdaptivePooling | 適応プーリング | output_size |

### B.4 正規化レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **Batch Normalization** | BatchNorm | バッチ正規化 | channels, eps |
| **Instance Normalization** | InstanceNorm | インスタンス正規化 | channels, eps |
| **Layer Normalization** | LayerNorm | レイヤー正規化 | affine |
| **Group Normalization** | GroupNorm | グループ正規化 | group, channels, eps |
| **Local Response Normalization** | LRN | 局所応答正規化 | region_type, local_size, alpha, beta |

### B.5 要素単位演算レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **加算** | Eltwise | 要素単位加算 | op_type=1 |
| **乗算** | Eltwise | 要素単位乗算 | op_type=0 |
| **最大値** | Eltwise | 要素単位最大値 | op_type=2 |
| **Binary演算** | BinaryOp | 二項演算 | op_type |
| **Unary演算** | UnaryOp | 単項演算 | op_type |

### B.6 形状操作レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **Reshape** | Reshape | 形状変更 | w, h, c |
| **Flatten** | Flatten | 平坦化 | - |
| **Permute** | Permute | 次元入れ替え | order_type |
| **Slice** | Slice | テンソル分割 | axis, slices |
| **Concat** | Concat | テンソル結合 | axis |
| **Split** | Split | テンソル分岐 | - |
| **Expand Dims** | ExpandDims | 次元拡張 | axes |
| **Squeeze** | Squeeze | 次元削除 | axes |

### B.7 アテンション・リカレントレイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **LSTM** | LSTM | LSTM層 | num_output, weight_data_size |
| **GRU** | GRU | GRU層 | num_output, weight_data_size |
| **RNN** | RNN | 基本RNN層 | num_output, weight_data_size |
| **MultiHead Attention** | MultiHeadAttention | マルチヘッドアテンション | embed_dim, num_heads |

### B.8 高度な演算レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **Softmax** | Softmax | ソフトマックス | axis |
| **Dropout** | Dropout | ドロップアウト | ratio |
| **Reduction** | Reduction | 次元削減演算 | operation, axes |
| **ArgMax** | ArgMax | 最大値インデックス | out_max_val, topk |
| **Clip** | Clip | 値クリッピング | min, max |
| **Pad** | Padding | パディング | top, bottom, left, right, type, value |

### B.9 データ変換レイヤー

| レイヤータイプ | ncnn名 | 説明 | パラメータ |
|--------------|-------|------|-----------|
| **Cast** | Cast | データ型変換 | type_from, type_to |
| **Quantize** | Quantize | 量子化 | scale |
| **Dequantize** | Dequantize | 逆量子化 | scale, bias |
| **Requantize** | Requantize | 再量子化 | scale_in, scale_out |

### B.10 カスタム・特殊レイヤー

| レイヤータイプ | ncnn名 | 説明 | 用途 |
|--------------|-------|------|------|
| **Input** | Input | 入力層 | ネットワーク入力 |
| **MemoryData** | MemoryData | メモリデータ | 定数データ |
| **Noop** | Noop | 何もしない | デバッグ用 |
| **Custom** | Custom | カスタム層 | ユーザー定義演算 |

## 付録C. 変換ツールオプション詳細

### C.1 onnx2ncnn オプション

```bash
# 基本構文
onnx2ncnn [options] model.onnx model.param model.bin

# 主要オプション
-o                  # 最適化を有効にする
-fp16               # FP16モードで出力
-int8               # INT8量子化を試行
-v                  # 詳細ログを出力
-shape-inference    # 形状推論を実行
```

**最適化オプション詳細**:
```bash
# 演算子融合の有効化
onnx2ncnn -o model.onnx model.param model.bin

# FP16変換（精度は若干低下、速度向上）
onnx2ncnn -fp16 model.onnx model.param model.bin

# デバッグ用詳細出力
onnx2ncnn -v model.onnx model.param model.bin 2>&1 | tee conversion.log
```

**対応ONNX演算子**:
- Conv, ConvTranspose
- BatchNormalization, InstanceNormalization
- Relu, LeakyRelu, Sigmoid, Tanh
- MaxPool, AveragePool, GlobalAveragePool
- Add, Mul, Sub, Div
- Reshape, Transpose, Squeeze, Unsqueeze
- Concat, Split, Slice
- MatMul, Gemm
- Softmax, LogSoftmax

### C.2 pnnx オプション

```bash
# PyTorchモデルの変換
pnnx model.pt [options]

# 主要オプション
inputshape=[1,3,224,224]    # 入力形状指定
inputshape2=[1,3,256,256]   # 複数入力形状（動的形状対応）
device=cpu                  # 実行デバイス
moduleop=                   # モジュール演算子リスト
optlevel=2                  # 最適化レベル（0-2）
```

**使用例**:
```bash
# 基本的な変換
pnnx resnet50.pt inputshape=[1,3,224,224]

# 動的バッチサイズ対応
pnnx model.pt inputshape=[1,3,224,224] inputshape2=[4,3,224,224]

# 最適化レベル指定
pnnx model.pt inputshape=[1,3,224,224] optlevel=2

# PNNX → ncnn変換
pnnx2ncnn model.pnnx.param model.pnnx.bin model.param model.bin
```

### C.3 ncnn2int8 オプション

```bash
# 基本的な量子化
ncnn2int8 model.param model.bin quantized.param quantized.bin images.txt

# オプション
-num-threads 4              # 使用スレッド数
-method 0                   # 量子化手法（0: KL散度, 1: 直接最小化）
-batch-size 1               # バッチサイズ
-progress                   # 進捗表示
```

**キャリブレーションファイル形式**:
```
# images.txt の例
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
...
```

**量子化手法の選択**:
- **method=0 (KL散度)**: 一般的により良い精度
- **method=1 (直接最小化)**: より高速、精度は若干劣る可能性

### C.4 ncnnoptimize オプション

```bash
# モデル最適化
ncnnoptimize model.param model.bin optimized.param optimized.bin [memory_budget]

# メモリバジェット指定（バイト単位）
ncnnoptimize model.param model.bin opt.param opt.bin 67108864  # 64MB

# 最適化の効果
# - 演算子融合（Convolution + BatchNorm + ReLU）
# - 冗長なレイヤーの除去
# - メモリレイアウトの最適化
```

## 付録D. サンプルデータセット

### D.1 画像分類用データセット

**CIFAR-10**:
```python
# PyTorchでのCIFAR-10読み込み例
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=False,
                          download=True, transform=transform)
```

**ImageNet 検証用サンプル**:
```bash
# ImageNet-1K から10クラス分のサンプル
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# 各クラス10枚ずつサンプリング
```

### D.2 物体検出用データセット

**COCO サンプル**:
```python
# MS COCO 2017 validation から抽出
import json

# アノテーションファイルから小規模サンプル作成
with open('instances_val2017.json', 'r') as f:
    coco_data = json.load(f)

# 人物クラス（category_id=1）のみ抽出
person_images = []
for ann in coco_data['annotations']:
    if ann['category_id'] == 1:
        person_images.append(ann['image_id'])

# 100枚に限定
sample_images = list(set(person_images))[:100]
```

### D.3 テスト用合成データ

**ランダム画像生成**:
```python
import numpy as np
import cv2

def generate_test_images(count=100, size=(224, 224)):
    """テスト用のランダム画像を生成"""
    images = []
    for i in range(count):
        # ランダムノイズ画像
        img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        # 簡単な幾何学パターンを追加
        center = (size[0]//2, size[1]//2)
        radius = np.random.randint(20, min(size)//3)
        color = (np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256))

        cv2.circle(img, center, radius, color, -1)

        # ファイル保存
        cv2.imwrite(f'test_image_{i:03d}.jpg', img)
        images.append(f'test_image_{i:03d}.jpg')

    return images

# テスト画像リストファイル作成
test_images = generate_test_images(100)
with open('test_images.txt', 'w') as f:
    for img in test_images:
        f.write(img + '\n')
```

### D.4 ベンチマーク用データセット

**パフォーマンステスト用**:
```python
# 異なるサイズの入力でのベンチマーク
import time
import numpy as np

def create_benchmark_data():
    """ベンチマーク用の様々なサイズのデータを作成"""
    sizes = [
        (224, 224),   # 標準的なサイズ
        (320, 320),   # 中程度のサイズ
        (416, 416),   # YOLO等で使用
        (512, 512),   # 高解像度
        (640, 640),   # さらに高解像度
    ]

    for size in sizes:
        img = np.random.rand(*size, 3).astype(np.float32)
        np.save(f'benchmark_{size[0]}x{size[1]}.npy', img)

        # 対応するテキストファイルも作成
        with open(f'benchmark_{size[0]}x{size[1]}.txt', 'w') as f:
            for i in range(10):  # 各サイズ10回測定用
                f.write(f'benchmark_{size[0]}x{size[1]}.npy\n')

create_benchmark_data()
```

**メモリ負荷テスト用**:
```python
def create_memory_stress_test():
    """メモリ使用量テスト用の大きなデータセット"""
    # 大量の小さな画像
    for i in range(1000):
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(f'memory_test/small_{i:04d}.jpg', img)

    # 少数の大きな画像
    for i in range(10):
        img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(f'memory_test/large_{i:02d}.jpg', img)

create_memory_stress_test()
```

---

以上で、ncnn Linux + C++チュートリアルの全12章と付録が完成しました。このチュートリアルは、ncnnの基礎から高度な応用まで、実践的な開発に必要な知識を包括的にカバーしています。

**チュートリアルの構成**:
1. **基礎編** (1-5章): ncnnの概要、環境構築、基本概念、モデル変換、基本API
2. **実装編** (6-7章): 画像分類と物体検出の具体的な実装
3. **最適化編** (8章): パフォーマンス最適化技術
4. **実践編** (9章): 実用的なアプリケーション開発
5. **応用編** (10章): カスタムレイヤー、マルチモデル、NEON最適化
6. **問題解決編** (11章): トラブルシューティング
7. **参考資料編** (12章): 公式ドキュメント、コミュニティリソース
8. **付録**: APIリファレンス、オペレーター一覧、ツールオプション、サンプルデータ

このチュートリアルを通じて、読者はncnnを使用した効率的で実用的なディープラーニングアプリケーションを開発するスキルを身につけることができます。