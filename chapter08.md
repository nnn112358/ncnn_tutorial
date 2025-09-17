# 第8章 カスタムレイヤー実装

## 8.1 カスタム化が必要となるケース
標準レイヤーで表現できない独自演算や、ONNX変換でサポート外となるオペレーションが含まれる場合は、ncnnにカスタムレイヤーを追加する必要がある。例えば特殊な正規化、Attentionモジュール、事前処理や後処理をモデル内に組み込んだパターンが該当する。また、性能面で既存レイヤーを置き換えて高速化したい場合にもカスタム実装が効果的だ。

## 8.2 Layer派生クラスの基本構造
カスタムレイヤーは`ncnn::Layer`を継承し、`load_param`、`load_model`、`forward`など必要なメソッドをオーバーライドする。以下はスカラー加算を行う単純な例である。
```cpp
class ScalarAdd : public ncnn::Layer {
public:
    ScalarAdd() { one_blob_only = true; }

    virtual int load_param(const ncnn::ParamDict& pd) {
        scalar = pd.get(0, 0.f);
        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom, ncnn::Mat& top,
                        const ncnn::Option& opt) const {
        top.create_like(bottom);
        for (int i = 0; i < bottom.total(); i++) {
            top[i] = bottom[i] + scalar;
        }
        return 0;
    }

private:
    float scalar;
};
```
`one_blob_only`や`support_inplace`などのフラグを設定し、レイヤーの入出力数やインプレース処理可否を宣言する。

## 8.3 レイヤー登録とParamファイル更新
実装したクラスは`DEFINE_LAYER_CREATOR(ScalarAdd)`マクロで生成関数を定義し、`Net::register_custom_layer("ScalarAdd", ScalarAdd_layer_creator);`などで登録する。`param`ファイルでは対象レイヤー名をカスタムレイヤー名に置き換え、必要なパラメータIDを指定する。ONNX変換時は`onnx2ncnn`に`--custom-op`オプションを渡し、該当ノードを自動的にカスタムレイヤーへマッピングするスクリプトを用意すると便利だ。

## 8.4 CPU最適化のポイント
CPU向けカスタムレイヤーでは、SIMD命令を活用することで大幅な高速化が見込める。ncnnはARM NEON用の`neon_mathfun.h`やx86 SSE/AVXヘルパーを用意しており、`ncnn::vfloat32x4`といったベクトル型を利用できる。メモリ配置を意識し、入力テンソルの連続性を前提にループ展開を行う。OpenMPを使ってチャネル単位で並列化する際は、スレッド数が多すぎると逆にオーバーヘッドになるためベンチマークで最適値を探る。

## 8.5 GPUシェーダーの実装
Vulkan対応を追加する場合、`forward_inplace`などGPU向けメソッドを実装し、SPIR-Vシェーダーを用意する。ncnnでは`src/layer/vulkan`ディレクトリに`.comp`ファイルとしてコンピュートシェーダーを書く。`ncnn::Layer::create_pipeline`でパイプラインを構築し、`destroy_pipeline`で解放する処理を用意する。シェーダー内では`shared`メモリを適切に使い、メモリアクセスを合致させる。CPUとGPUの結果が一致するか、`vkdev->option.use_fp16_storage`などの設定で挙動が変わらないかを確認する。

## 8.6 モデル変換との連携
学習フレームワークからONNXへエクスポートする際、カスタムレイヤーのノードを保持するための工夫が必要である。PyTorchでは`symbolic`定義を追加し、ONNXへのマッピングを提供する。`onnx-simplifier`を通す場合はノードが削除されないよう除外設定を行う。ncnn側では`customop.h`にオペコードを追加し、`ncnnoptimize`や量子化ツールがカスタムレイヤーをスキップせずに処理できるよう更新する。

## 8.7 テストと検証
カスタムレイヤーはユニットテストとベンチマークをセットで実施する。浮動小数点誤差を検証するために、学習フレームワーク側の演算結果とncnn実装を比較する。INT8対応が必要な場合は、量子化テーブルにエントリを追加し、スケールの適用が正しいか確認する。CIではテストモデルを用意し、`Extractor::extract`で出力をチェックする自動テストを組み込む。Vulkan対応の場合はCPU・GPU両方の出力差分を取る。

## 8.8 本章のまとめ
本章ではncnnにカスタムレイヤーを追加する際の手順を、実装・登録・最適化・モデル変換との統合・テストまで一貫して解説した。次章では実プロジェクトでの適用例を紹介し、カスタムレイヤーや最適化をどのように活用したかをケーススタディ形式で見ていく。
