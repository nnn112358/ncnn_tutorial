# 第5章 APIを使った推論アプリ開発

## 5.1 最小構成の推論フロー
ncnnアプリケーションは基本的に次の手順で構成される。(1) `Net`オブジェクトの生成と`param`/`bin`読み込み、(2) `Extractor`の生成、(3) 入力データの前処理と`Mat`への格納、(4) 推論実行、(5) 後処理と結果表示。この流れを理解しておけば、画像分類・物体検出・セグメンテーションなど用途が変わっても共通の骨組みを再利用できる。

## 5.2 C++による基本コード例
以下はC++でSqueezeNetを推論する最小サンプルである。
```cpp
#include "net.h"

int main() {
    ncnn::Net net;
    net.opt.use_vulkan_compute = true; // Vulkan対応端末ならGPUを利用

    net.load_param("squeezenet_v1.1.param");
    net.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        image_data, ncnn::Mat::PIXEL_BGR, width, height, 227, 227);
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    const float norm_vals[3] = {1.f/58.f, 1.f/58.f, 1.f/58.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    // 後処理: 最も確率の高いクラスを表示
    int top_class = out.channel(0)[0].a;
    printf("top class: %d\n", top_class);
}
```
画像データはOpenCVなどで読み込み、ピクセルフォーマットを`Mat::PIXEL_`列挙子に合わせる。`substract_mean_normalize`で平均値・標準化を適用し、学習時と同じスケーリングを再現する。

## 5.3 AndroidでのJNIブリッジ
Androidアプリでは、Java/Kotlin層とC++層をJNIで接続する。`native-lib.cpp`でncnnを利用する関数を実装し、`CMakeLists.txt`で`ncnn`ライブラリをリンクする。
```cpp
extern "C" JNIEXPORT jint JNICALL
Java_com_example_app_NcnnWrapper_detect(JNIEnv* env, jobject thiz, jobject bitmap) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);

    void* pixels = nullptr;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    ncnn::Mat in = ncnn::Mat::from_pixels(
        (const unsigned char*)pixels, ncnn::Mat::PIXEL_RGBA,
        info.width, info.height);
    AndroidBitmap_unlockPixels(env, bitmap);

    // Net生成・推論処理（シングルトンで再利用）
    auto& net = NcnnSingleton::instance();
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);
    return postprocess(out);
}
```
Java側では`System.loadLibrary("ncnn")`で共有ライブラリを読み込み、カメラプレビューなどの画像をJNI経由で渡す。マルチスレッドで同じ`Net`を共有する場合は排他制御やExtractorの再利用戦略を検討する。

## 5.4 画像前処理・後処理の実装
前処理では入力解像度へのリサイズ、チャネル順の変換、正規化が必須である。ncnnの`Mat::from_pixels_resize`がリサイズと色変換を同時にこなすため効率が良い。後処理はタスクによって異なり、分類ではsoftmax・Top-K抽出、検出ではAnchor生成とNon-Maximum Suppression (NMS)、セグメンテーションではマスク生成とブレンディングを行う。ncnnには`ncnn::LayerType::Softmax`や`ncnn::LayerType::DetectionOutput`など一部の後処理レイヤーも実装されているが、アプリ側で独自ロジックを組み込むことも多い。

## 5.5 モバイルUIとの連携
Androidでは`TextureView`や`SurfaceView`に推論結果を描画し、iOSでは`MetalKit`や`UIKit`の描画APIを使用する。推論スレッドはUIスレッドと分離し、結果はスレッドセーフなキューやLiveData、Combineなどで渡す。フレーム単位の推論が必要な場合は、FPSとレイテンシを計測し、`Net::opt`で使用するスレッド数やVulkan利用の有無を調整する。省電力モードではバッチ間隔を伸ばす、解像度を下げるなどの制御も検討する。

## 5.6 ロギングとエラーハンドリング
ncnnは`Net::load_param`や`Extractor::extract`が失敗した場合、エラーコードを返す。これらをチェックしてユーザーに適切なメッセージを表示する。推論結果の異常やVulkanの初期化失敗を検知するため、`extractor.set_num_threads()`や`vkdev->info`などでハードウェア情報をログ出力する。クラッシュレポートと組み合わせて、端末依存の問題を早期に把握する体制を整えると運用が安定する。

## 5.7 バッチ処理とストリーミング
サーバーサイドやエッジゲートウェイでは、複数入力をバッチ処理してスループットを高める場合がある。ncnnは動的バッチに制限があるため、複数の`Extractor`を並列起動し、入力ごとに独立した推論を行う設計が推奨される。一方、動画ストリーミング処理ではリングバッファを用意し、最新フレームを常に保持しながら推論結果を順次出力する。パイプライン全体を非同期化し、前処理→推論→後処理の各段階でキューを挟むとCPU・GPU資源の利用効率が向上する。

## 5.8 本章のまとめ
本章ではncnnのAPIを利用した推論アプリを開発する手順を、C++の基本例からAndroidのJNI連携、前処理・後処理、UI統合、ロギング、並列処理まで幅広く解説した。次章ではモバイル・エッジ端末での最適化手法に焦点を当て、実際のプロファイリング結果をもとにチューニング戦略を紹介する。
