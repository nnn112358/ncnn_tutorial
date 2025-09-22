# 2. 環境構築

## 2.1 Linux環境の準備

ncnnの開発を始める前に、適切なLinux環境を準備する必要があります。本チュートリアルでは、Ubuntu 20.04 LTS以降を推奨しますが、CentOS 8、Fedora 32以降、Debian 10以降でも同様の手順で環境構築が可能です。

### システム要件の確認

まず、システムが最小要件を満たしているか確認しましょう：

```bash
# CPUアーキテクチャの確認
uname -m

# メモリ容量の確認
free -h

# ディスク容量の確認
df -h

# OSバージョンの確認
cat /etc/os-release
```

### パッケージマネージャーの更新

環境構築を始める前に、パッケージマネージャーを最新状態に更新します：

```bash
# Ubuntu/Debianの場合
sudo apt update && sudo apt upgrade -y

# CentOS/RHELの場合
sudo yum update -y

# Fedoraの場合
sudo dnf update -y
```

## 2.2 必要な依存関係のインストール

ncnnのビルドには複数の依存関係が必要です。以下の手順に従って、必要なパッケージをインストールしていきます。

### 2.2.1 CMake

CMakeはncnnのビルドシステムで使用されます。バージョン3.10以降が必要です：

```bash
# Ubuntu/Debianの場合
sudo apt install cmake

# CentOS/RHELの場合
sudo yum install cmake

# Fedoraの場合
sudo dnf install cmake

# バージョン確認
cmake --version
```

CMakeのバージョンが古い場合は、公式サイトから最新版をダウンロードしてインストールします：

```bash
# 最新版CMakeのダウンロード（例：3.25.0）
wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.tar.gz
tar -xzf cmake-3.25.0-linux-x86_64.tar.gz
sudo mv cmake-3.25.0-linux-x86_64 /opt/cmake
sudo ln -sf /opt/cmake/bin/* /usr/local/bin/
```

### 2.2.2 Protocol Buffers

Protocol Buffersは、一部のモデル変換ツールで使用されます：

```bash
# Ubuntu/Debianの場合
sudo apt install libprotobuf-dev protobuf-compiler

# CentOS/RHELの場合
sudo yum install protobuf-devel protobuf-compiler

# Fedoraの場合
sudo dnf install protobuf-devel protobuf-compiler

# インストール確認
protoc --version
```

### 2.2.3 Vulkan SDK（オプション）

GPU加速を利用したい場合は、Vulkan SDKをインストールします：

```bash
# Ubuntu/Debianの場合
sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev

# パッケージが見つからない場合は、LunarG公式リポジトリを追加
wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
sudo apt update
sudo apt install vulkan-sdk

# インストール確認
vulkaninfo
```

### その他の必要なツール

開発に必要なその他のツールもインストールします：

```bash
# Ubuntu/Debianの場合
sudo apt install build-essential git wget curl unzip

# CentOS/RHELの場合
sudo yum groupinstall "Development Tools"
sudo yum install git wget curl unzip

# Fedoraの場合
sudo dnf groupinstall "Development Tools"
sudo dnf install git wget curl unzip
```

## 2.3 ncnnライブラリのビルド

### 2.3.1 ソースコードの取得

GitHubからncnnのソースコードをクローンします：

```bash
# ホームディレクトリに移動
cd ~

# ncnnリポジトリのクローン
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# サブモジュールの初期化（重要）
git submodule update --init
```

### 2.3.2 ビルド設定

CMakeを使用してビルド設定を行います。様々なオプションを指定して、用途に応じた最適化を行えます：

```bash
# ビルドディレクトリの作成
mkdir build
cd build

# 基本的なビルド設定
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=ON \
      -DNCNN_BUILD_EXAMPLES=ON \
      -DNCNN_BUILD_TOOLS=ON \
      -DNCNN_BUILD_BENCHMARK=ON \
      ..
```

### 主要なCMakeオプション

| オプション | 説明 | 推奨値 |
|-----------|------|--------|
| `CMAKE_BUILD_TYPE` | ビルドタイプ | `Release` |
| `NCNN_VULKAN` | Vulkan GPU加速 | `ON` |
| `NCNN_BUILD_EXAMPLES` | サンプルコードのビルド | `ON` |
| `NCNN_BUILD_TOOLS` | 変換ツールのビルド | `ON` |
| `NCNN_BUILD_BENCHMARK` | ベンチマークツールのビルド | `ON` |
| `NCNN_OPENMP` | OpenMP並列処理 | `ON` |
| `NCNN_THREADS` | スレッドサポート | `ON` |

### 高度なビルド設定

特定の用途に応じて、より詳細な設定を行うことができます：

```bash
# ARM NEON最適化を有効にした設定（ARM環境の場合）
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=ON \
      -DNCNN_ARM82=ON \
      -DNCNN_BUILD_EXAMPLES=ON \
      -DNCNN_BUILD_TOOLS=ON \
      ..

# デバッグビルド設定
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=ON \
      -DNCNN_BUILD_TOOLS=ON \
      ..
```

### 2.3.3 コンパイル実行

設定が完了したら、実際にコンパイルを実行します：

```bash
# 並列コンパイルの実行（CPUコア数に応じて-jオプションを調整）
make -j$(nproc)

# または、より詳細な出力でコンパイル
make -j$(nproc) VERBOSE=1
```

コンパイルには環境によって5～30分程度かかります。エラーが発生した場合は、依存関係の不足や設定の問題が考えられます。

## 2.4 インストールの確認

ビルドが正常に完了したら、動作確認を行います：

```bash
# ビルド成果物の確認
ls -la src/

# サンプルプログラムの実行
cd examples
./squeezenet ../images/256-ncnn.png

# ベンチマークの実行
cd ../benchmark
./benchncnn 10 4 0 -1 1
```

### インストール（オプション）

システム全体でncnnを使用したい場合は、インストールを行います：

```bash
# インストールの実行
sudo make install

# インストール場所の確認
ls -la /usr/local/include/ncnn/
ls -la /usr/local/lib/libncnn*
```

### 環境変数の設定

ncnnを便利に使用するために、環境変数を設定します：

```bash
# .bashrcまたは.zshrcに追加
echo 'export NCNN_ROOT=~/ncnn' >> ~/.bashrc
echo 'export PATH=$NCNN_ROOT/build/tools:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NCNN_ROOT/build/src:$LD_LIBRARY_PATH' >> ~/.bashrc

# 環境変数の読み込み
source ~/.bashrc
```

## 2.5 Python環境の構築

ncnnには公式のPythonバインディングも提供されており、プロトタイピングやテストに便利です。

### 2.5.1 Python 3.xのインストール

まず、Python 3.7以降がインストールされていることを確認します：

```bash
# Pythonバージョンの確認
python3 --version

# pipの確認
python3 -m pip --version

# 必要に応じてPythonをインストール
# Ubuntu/Debianの場合
sudo apt install python3 python3-pip python3-dev

# CentOS/RHELの場合
sudo yum install python3 python3-pip python3-devel

# Fedoraの場合
sudo dnf install python3 python3-pip python3-devel
```

### 2.5.2 ncnn-pythonパッケージのインストール

公式のPythonパッケージをインストールします：

```bash
# pipを使用したインストール
python3 -m pip install --user ncnn

# または、開発版をソースからビルド
cd ~/ncnn/python
python3 setup.py build_ext --inplace
python3 -m pip install --user .
```

### 2.5.3 仮想環境の設定

プロジェクトごとに環境を分離するため、仮想環境の使用を推奨します：

```bash
# venvによる仮想環境の作成
python3 -m venv ncnn-env

# 仮想環境の有効化
source ncnn-env/bin/activate

# ncnnとその他の依存パッケージのインストール
pip install ncnn numpy opencv-python matplotlib

# Jupyter Notebookも使用する場合
pip install jupyter notebook
```

### Python環境の動作確認

インストールが正常に完了したか確認します：

```python
# Pythonスクリプトで動作確認
python3 -c "
import ncnn
print('ncnn version:', ncnn.__version__)

# 簡単な動作テスト
net = ncnn.Net()
print('ncnn.Net created successfully')
"
```

これで、ncnnの開発環境が完全に構築されました。次章では、ncnnの基本概念について詳しく学習していきます。

### トラブルシューティング

環境構築の過程では、システム環境の違いや依存関係の問題により、様々なエラーが発生する可能性があります。ここでは、特に頻繁に遭遇する問題とその解決方法について説明します。

CMakeに関連するエラーは最も一般的な問題の一つです。古いバージョンのCMakeがインストールされている場合、ncnnの最新機能を利用できないことがあります。この場合は、CMakeの公式サイトから最新版をダウンロードして手動インストールするか、パッケージマネージャーを通じてアップグレードを行ってください。また、必要な依存パッケージが不足している場合も、CMakeの設定段階でエラーが発生します。エラーメッセージを注意深く確認し、不足しているライブラリやツールを追加でインストールすることで解決できます。

コンパイル時のエラーについては、GCCやClangのバージョンが古い場合に、C++の新しい機能をサポートしていないことが原因となることがあります。コンパイラを最新版にアップグレードするか、ncnnが要求する最小バージョン以上のものを使用してください。また、システムのメモリが不足している場合、並列コンパイル時にシステムが不安定になることがあります。このような場合は、`make -j1`のように並列度を下げるか、スワップファイルを増設することで解決できます。

Vulkan関連のエラーは、GPU加速機能を有効にしている場合に発生することがあります。まず、使用しているグラフィックドライバーが最新であることを確認してください。古いドライバーではVulkanの最新機能がサポートされていない可能性があります。また、そもそもVulkanに対応していないグラフィックスハードウェアを使用している場合は、CMakeの設定で`-DNCNN_VULKAN=OFF`オプションを指定してVulkan機能を無効化してください。