# 4. モデル変換

## 4.1 サポートされるフレームワーク

ncnnは、主要な深層学習フレームワークで学習されたモデルを変換するための包括的なツールセットを提供しています。以下のフレームワークからの変換が公式にサポートされています：

### 対応フレームワーク一覧

| フレームワーク | 変換ツール | 推奨度 | 備考 |
|---------------|-----------|--------|------|
| **ONNX** | `onnx2ncnn` | ★★★★★ | 最も安定、多くのモデルで動作確認済み |
| **PyTorch** | `pnnx` + `pnnx2ncnn` | ★★★★☆ | 公式推奨、最新モデルに対応 |
| **TensorFlow** | `tf2onnx` + `onnx2ncnn` | ★★★☆☆ | ONNX経由での変換を推奨 |
| **Caffe** | `caffe2ncnn` | ★★☆☆☆ | レガシーサポート |
| **MXNet** | `mxnet2ncnn` | ★★☆☆☆ | 限定的サポート |
| **Darknet** | `darknet2ncnn` | ★★☆☆☆ | YOLO等の特定モデル向け |

### 変換フローの概要

```
学習済みモデル → 中間形式（ONNX推奨） → ncnn形式（.param + .bin）
```

ONNXを中間形式として使用することで、多くのフレームワークからの変換が可能になり、変換プロセスの標準化と安定性が向上します。

## 4.2 ONNXからncnnへの変換

ONNXは現在最も推奨される変換パスです。多くのフレームワークがONNX出力をサポートしており、ncnnでも最も安定した変換結果が得られます。

### 4.2.1 onnx2ncnnツールの使用

**基本的な変換コマンド**:
```bash
# 基本的な変換
onnx2ncnn model.onnx model.param model.bin

# 最適化オプション付きの変換
onnx2ncnn model.onnx model.param model.bin -o
```

### 4.2.2 変換オプション

**よく使用されるオプション**:

| オプション | 説明 | 使用例 |
|-----------|------|--------|
| `-o` | モデル最適化を有効にする | `onnx2ncnn -o model.onnx out.param out.bin` |
| `-fp16` | 16bit浮動小数点数で出力 | `onnx2ncnn -fp16 model.onnx out.param out.bin` |

**詳細な変換例**:

```bash
# ResNet-50の変換例
# 1. PyTorchからONNXへエクスポート（Python）
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet50.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                               'output': {0: 'batch_size'}})

# 2. ONNXからncnnへ変換
onnx2ncnn resnet50.onnx resnet50.param resnet50.bin -o
```

### 変換時の注意点

**入力形状の確認**:
```bash
# ONNXモデルの詳細情報を確認
pip install onnx
python -c "
import onnx
model = onnx.load('model.onnx')
print('Inputs:')
for input in model.graph.input:
    print(f'  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}')
print('Outputs:')
for output in model.graph.output:
    print(f'  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}')
"
```

**サポートされていない演算子の確認**:
```bash
# 変換ログを詳細出力モードで確認
onnx2ncnn model.onnx model.param model.bin -v
```

## 4.3 PyTorchからncnnへの変換

PyTorchからの変換には、公式の`pnnx`（PyTorch Neural Network eXchange）ツールを使用します。

### pnnxツールのビルド

```bash
# pnnxのビルド（ncnnソースディレクトリ内で）
cd tools/pnnx
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 変換プロセス

**1. PyTorchモデルの準備**:
```python
import torch
import torchvision.models as models

# モデルの準備
model = models.mobilenet_v2(pretrained=True)
model.eval()

# トレースの作成
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)

# pnnx形式で保存
traced_model.save('mobilenet_v2.pt')
```

**2. pnnxによる変換**:
```bash
# PyTorchからpnnx形式への変換
./pnnx mobilenet_v2.pt inputshape=[1,3,224,224]

# pnnxからncnn形式への変換
pnnx2ncnn mobilenet_v2.pnnx.param mobilenet_v2.pnnx.bin mobilenet_v2.param mobilenet_v2.bin
```

### 高度な変換オプション

**動的形状への対応**:
```bash
# 動的バッチサイズ対応
./pnnx model.pt inputshape=[1,3,224,224] inputshape2=[4,3,224,224]
```

**カスタム演算子の処理**:
```python
# カスタム演算子の登録
@torch.jit.script
def custom_op(x):
    return torch.relu(x) * 2

model.custom_layer = custom_op
```

## 4.4 TensorFlowからncnnへの変換

TensorFlowモデルは、まずONNX形式に変換してからncnnに変換するのが推奨される方法です。

### tf2onnxを使用した変換

**1. 必要なパッケージのインストール**:
```bash
pip install tf2onnx tensorflow onnx
```

**2. SavedModel形式からONNXへの変換**:
```bash
# SavedModelからONNXへ
python -m tf2onnx.convert --saved-model tensorflow_model_dir --output model.onnx

# Frozen graphからONNXへ
python -m tf2onnx.convert --input frozen_graph.pb --inputs input:0 --outputs output:0 --output model.onnx
```

**3. Keras (.h5) モデルからの変換**:
```python
import tensorflow as tf
from tf2onnx import convert

# Kerasモデルの読み込み
model = tf.keras.models.load_model('model.h5')

# ONNX形式で保存
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = convert.from_keras(model, input_signature=spec, opset=13)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

**4. ONNXからncnnへの変換**:
```bash
onnx2ncnn model.onnx model.param model.bin -o
```

## 4.5 Caffeからncnnへの変換

Caffeモデルは直接ncnn形式に変換できますが、現在は主にレガシーサポートとなっています。

### 基本的な変換

```bash
# Caffeからncnnへの直接変換
caffe2ncnn deploy.prototxt model.caffemodel model.param model.bin

# 最適化オプション付き
caffe2ncnn deploy.prototxt model.caffemodel model.param model.bin -O
```

### Caffe変換の制限事項

Caffeからncnnへの変換には、いくつかの技術的制約があります。新しく開発されたレイヤータイプについては、ncnnの変換ツールが対応していない場合があり、特に最新の研究で提案された演算子や独自に実装されたレイヤーでは変換エラーが発生する可能性があります。また、Caffeで独自に定義されたカスタムレイヤーの変換は技術的に困難な場合が多く、手動でのncnn対応レイヤー実装が必要になることがあります。さらに、Caffeフレームワーク自体の開発が終了に近い状況であることから、caffe2ncnn変換ツールのメンテナンスも限定的となっており、新しい問題への対応は期待できない状況です。

## 4.6 量子化（8bit）

量子化により、モデルサイズの削減と推論速度の向上が可能です。

### 量子化の種類

**1. ポスト量子化（Post-training Quantization）**:
```bash
# ncnn2intツールを使用した量子化
ncnn2int8 model.param model.bin quantized_model.param quantized_model.bin calibration_images.txt
```

**2. 量子化対応学習（Quantization Aware Training）**:
学習時から量子化を考慮したモデルを作成し、より高精度な量子化モデルを得る方法。

### 量子化の実行手順

**1. キャリブレーションデータの準備**:
```bash
# 代表的な入力画像のリストを作成
find ./calibration_images -name "*.jpg" > calibration_list.txt
```

**2. 量子化の実行**:
```bash
# 8bit量子化の実行
ncnn2int8 original.param original.bin quantized.param quantized.bin calibration_list.txt
```

**3. 量子化結果の確認**:
```bash
# モデルサイズの比較
ls -lh original.bin quantized.bin

# 精度の確認（ベンチマークツールを使用）
./benchncnn 10 1 0 0 0 original.param original.bin
./benchncnn 10 1 0 0 0 quantized.param quantized.bin
```

### 量子化のベストプラクティス

**キャリブレーションデータの選択**について、適切なデータ選択は量子化の成功に直結する重要な要素です。実際の運用環境で使用されるデータと統計的分布が類似したデータを選択することで、量子化後の精度低下を最小限に抑えることができます。データセットのサイズについては、通常1000から5000サンプル程度の十分な多様性を持つデータが推奨されており、これにより様々な入力パターンに対する量子化パラメータの最適化が可能になります。また、学習時やオリジナルモデルでの推論時と同様の前処理（正規化、リサイズ、色空間変換等）を適用したデータを使用することで、量子化プロセスの精度向上が期待できます。

**量子化後の精度検証**:
```python
# Python での精度比較例
import numpy as np

def compare_accuracy(original_net, quantized_net, test_data):
    original_results = []
    quantized_results = []

    for data in test_data:
        # オリジナルモデルでの推論
        original_out = original_net.forward(data)
        original_results.append(original_out)

        # 量子化モデルでの推論
        quantized_out = quantized_net.forward(data)
        quantized_results.append(quantized_out)

    # 精度の比較
    mse = np.mean((np.array(original_results) - np.array(quantized_results))**2)
    print(f"MSE between original and quantized: {mse}")
```

### トラブルシューティング

**変換エラーの対処法**について、モデル変換プロセスで遭遇する一般的な問題と解決策を説明します。

**サポートされていない演算子**の問題は、変換ツールが特定のONNX演算子やレイヤータイプに対応していない場合に発生します。この問題を特定するには、詳細なエラーログを確認することが有効です：

```bash
# エラーログの確認
onnx2ncnn model.onnx out.param out.bin -v 2>&1 | grep "not supported"
```

**形状推論エラー**は、動的な入力形状や不明確な次元定義が原因で発生することが多い問題です。この場合、ONNXモデルの入力と出力の形状を明示的に固定することで解決できます：

```python
# ONNX モデルの形状を固定
import onnx
from onnx.tools import update_model_dims

model = onnx.load('model.onnx')
updated_model = update_model_dims.update_inputs_outputs_dims(
    model, {'input': [1, 3, 224, 224]}, {'output': [1, 1000]}
)
onnx.save(updated_model, 'model_fixed.onnx')
```

**メモリ不足**エラーは、大規模なモデルを変換する際に発生する可能性があります。この問題は、バッチサイズを制限することで回避できる場合があります：

```bash
# より小さなバッチサイズで変換
onnx2ncnn --max-batch-size 1 model.onnx out.param out.bin
```

モデル変換は、ncnnを使用する上で重要なプロセスです。適切な変換パスと最適化オプションを選択することで、高性能なモバイル推論アプリケーションを開発することができます。次章では、変換されたモデルを実際にC++で使用する方法について詳しく学習します。