# ファインチューニング

PP-OCRv5 認識モデルをファインチューニングして、特定の文字の精度を改善します。 パイプラインは4ステップで構成されます：データ生成（Rust）→ 訓練（Python/PaddleOCR）→ ONNX エクスポート → モデル差し替え

> **注意**: 訓練には **Python 3.12** が必要です（PaddlePaddle 3.3 は 3.13+ 非対応）。`scripts/` ディレクトリは [uv](https://docs.astral.sh/uv/) で Python 環境を管理しています。

### 前提条件

```bash
# PaddleOCR をクローン（訓練スクリプト用）
git clone https://github.com/PaddlePaddle/PaddleOCR.git /tmp/PaddleOCR

# Python 訓練依存パッケージをインストール
uv sync --project scripts --extra train
```

### ステップ 1: 訓練データの生成

`ocrus-dataset` クレートがテキスト画像を生成し、オーグメンテーション（回転、ぼかし、ノイズ、コントラスト）を適用します。データ生成は Rust + rayon 並列で実行されます。

2つの方法があります：
- **`--test-images`（推奨）**: `test_images/` に事前生成された画像を使用。フォントインストール不要、クロスプラットフォーム対応
- **フォント再レンダリング**: システムフォントからリアルタイムにレンダリング。`--all-fonts` で全利用可能フォントを使用

```bash
# 対象文字カテゴリの訓練データを生成（フォントからレンダリング）
ocrus dataset generate \
  --output /tmp/ocrus_training_data \
  --categories hiragana,katakana,halfwidth_alnum,fullwidth_alnum \
  --samples-per-char 5

# テスト失敗結果から重点的にデータを生成（事前生成済み画像を使用、推奨）
ocrus dataset from-failures \
  --failures ./test_results/failures_step1.json \
  --test-images ./test_images \
  --output /tmp/ocrus_training_data \
  --samples-per-char 10

# テスト失敗結果からフォント再レンダリングで生成
ocrus dataset from-failures \
  --failures ./test_results/failures_step1.json \
  --output /tmp/ocrus_training_data \
  --samples-per-char 10 --all-fonts
```

文字カテゴリ一覧：

| カテゴリ | 内容 | 文字数 |
|----------|------|--------|
| `halfwidth_alnum` | 半角英数字（A-Z, a-z, 0-9） | 62 |
| `halfwidth_symbols` | 半角記号（!@#$%&... 等） | 32 |
| `fullwidth_alnum` | 全角英数字（Ａ-Ｚ, ａ-ｚ, ０-９） | 62 |
| `fullwidth_symbols` | 全角記号・日本語句読点（、。「」…） | 63 |
| `hiragana` | ひらがな（あ-ん） | 83 |
| `katakana` | カタカナ（ア-ヶ） | 86 |
| `joyo_kanji` | 常用漢字（2010年改定） | 2,136 |
| `jis_level1` | JIS X 0208 第1水準漢字 | 2,965 |
| `jis_level2` | JIS X 0208 第2水準漢字 | 3,390 |
| `jis_level3` | JIS X 0213 第3水準漢字 | 1,233 |
| `jis_level4` | JIS X 0213 第4水準漢字 | 7,960 |

フォントスタイル（`--font-styles` オプション）：

| スタイル | 説明 | マッチパターン |
|----------|------|----------------|
| `mincho` | 明朝体 | mincho, 明朝, serif, song, batang |
| `gothic` | ゴシック体 | gothic, ゴシック, sans, kaku, maru |
| `script` | 筆書体 | script, brush, 筆, gyosho, kaisho |
| `monospace` | 等幅 | mono, courier, consolas, menlo |
| `other` | その他 | （デフォルト） |

出力フォーマット：
```text
/tmp/ocrus_training_data/
  manifest.json      # メタデータ（フォント、カテゴリ、オーグメント設定）
  labels.tsv         # ファイル名 \t 正解 \t カテゴリ \t フォント \t オーグメント
  samples/           # レンダリング済み PNG 画像（高さ 48px）
    000000.png
    000001.png
    ...
```

`labels.tsv` と `samples/` が生成されます。PaddleOCR 用の `train.txt` / `val.txt` は `finetune` コマンドが変換時に作成します。`tools/train.py` を直接使う場合は、 訓練・検証リストを別途用意してください。

### ステップ 2: 事前学習済み重みのダウンロード

```bash
# PP-OCRv5 server rec 事前学習済み重みをダウンロード（約214MB）
mkdir -p models/pretrained
curl -L -o models/pretrained/PP-OCRv5_server_rec_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

### ステップ 3: PaddleOCR でファインチューン

#### CPU vs GPU

| | CPU | GPU（例: RTX 4060 Super） |
|---|---|---|
| PaddlePaddle | `paddlepaddle-gpu==3.3.0`（CPU 実行） | `paddlepaddle-gpu==3.3.0` |
| 設定: `use_gpu` | `false` | `true` |
| 設定: `batch_size_per_card` | 32 | 128 |
| 設定: `num_workers` | 0 | 4-8 |
| 速度 | 約18秒/バッチ | 約0.4秒/バッチ |
| 5エポック（276k画像） | 約8日 | 約4時間 |

```bash
# 訓練依存（Linux の CUDA 配布版。CPU 実行時は use_gpu: false）
uv sync --project scripts --extra train
```

#### 訓練設定

YAML 設定ファイルを作成します（`PP-OCRv5_server_rec.yml` ベース）。以下は CPU の例です。GPU の場合は上記の表に従って `use_gpu`、`batch_size_per_card`、`num_workers` を変更してください。

```yaml
Global:
  model_name: PP-OCRv5_server_rec
  use_gpu: false
  epoch_num: 5
  save_model_dir: /tmp/ocrus_finetune_output
  pretrained_model: ./models/pretrained/PP-OCRv5_server_rec_pretrained
  character_dict_path: /tmp/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt
  max_text_length: &max_text_length 25
  eval_batch_step: [500, 1000]

Optimizer:
  name: Adam
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 1

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /tmp/ocrus_training_data/
    label_file_list:
    - /tmp/ocrus_training_data/train_list.txt
  loader:
    batch_size_per_card: 32
    num_workers: 0
```

完全な設定リファレンスは `PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` を参照してください。

#### 訓練の実行

```bash
PYTHONPATH=/tmp/PaddleOCR:$PYTHONPATH \
  uv run --project scripts --python 3.12 python3 -u /tmp/PaddleOCR/tools/train.py \
  -c /path/to/your_config.yml
```

訓練出力：
```text
/tmp/ocrus_finetune_output/
  train.log              # 訓練ログ
  config.yml             # 保存された設定
  best_accuracy/         # 最良モデルのチェックポイント
    best_accuracy.pdparams
  latest/                # 最新チェックポイント（再開用）
```

チェックポイントから再開するには、設定に追加：
```yaml
Global:
  checkpoints: /tmp/ocrus_finetune_output/latest
```

### ステップ 4: ONNX にエクスポート

```bash
# 最良モデルを ONNX にエクスポート
uv run --project scripts export-onnx \
  --model /tmp/ocrus_finetune_output/best_accuracy \
  --output rec_finetuned.onnx

# デフォルトモデルとしてインストール
uv run --project scripts export-onnx \
  --model /tmp/ocrus_finetune_output/best_accuracy \
  --output rec_finetuned.onnx \
  --install   # ~/.ocrus/models/rec.onnx にコピー

# .ocnn フォーマットに変換
uv run --project scripts --extra convert python scripts/src/ocrus_scripts/convert_to_ocnn.py \
  rec_finetuned.onnx -o ~/.ocrus/models/rec.ocnn
```

### ステップ 5（任意）: INT8 量子化

```bash
uv sync --project scripts --extra quantize

uv run --project scripts quantize \
  --input rec_finetuned.onnx \
  --output rec_int8.onnx
```

### 精度テスト

モデルの評価と弱い文字の特定のため、文字精度テストを実行します。 段階的にテストを進められるよう、ステップごとにテストを分割しています：

| ステップ | 対象 | 文字数 |
|----------|------|--------|
| `step1` | 半角/全角 英数記号 | ~220 |
| `step2` | ひらがな・カタカナ | ~170 |
| `step3_joyo` | 常用漢字 | 2,136 |
| `step3_jis1` | JIS 第1水準漢字 | 2,965 |
| `step3_jis2` | JIS 第2水準漢字 | 3,390 |
| `step3_jis3` | JIS 第3水準漢字 | 1,233 |
| `step3_jis4` | JIS 第4水準漢字 | 7,960 |

```bash
# Step 1: 半角/全角 英数記号（約36分、環境によって変動）
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture

# Step 2: ひらがな・カタカナ
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture

# Step 3: 漢字（水準ごとに個別実行）
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_joyo --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis1 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis2 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis3 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_jis4 --release -- --ignored --nocapture

# 全ステップ一括
cargo test -p ocrus-cli --test char_accuracy char_accuracy_all --release -- --ignored --nocapture

# 量子化モデルとの A/B テスト（任意のステップと組み合わせ可能）
OCRUS_QUANTIZED_MODEL=rec_int8.onnx \
  cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
```

テスト結果の出力先：
- `logs/char_accuracy_{step}_{timestamp}.log` — テストログ（フォント/カテゴリごとの精度、処理速度、ETA）
- `test_results/failures_{step}.json` — 失敗した文字（ステップ 1 の `from-failures` にフィードバック可能）

失敗文字はカテゴリ完了ごとに逐次保存されます。Ctrl+C で中断した場合もその時点までの結果が保存されます。
