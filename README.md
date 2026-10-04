<p align="center">
  <img src="docs/images/app.png" width="128" alt="OCRus">
</p>

<h1 align="center">OCRus</h1>

<p align="center">
  純 Rust の推論エンジンと CLI・Python バインディングを備えた日本語 OCR
</p>

<!-- standard:badges:start -->
<h3 align="center">対応プラットフォーム</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&amp;logoColor=black" alt="Linux">
  <img src="https://img.shields.io/badge/macOS-000000?logo=apple&amp;logoColor=white" alt="macOS">
</p>

<p align="center">
  <a href="https://github.com/owayo/ocrus/actions/workflows/ci.yml"><img src="https://github.com/owayo/ocrus/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/owayo/ocrus/releases/latest"><img src="https://img.shields.io/github/v/release/owayo/ocrus" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/owayo/ocrus" alt="License"></a>
</p>
<!-- standard:badges:end -->

---

## 機能

- **SIMD 前処理**: グレースケール変換、二値化、正規化、射影を高速化します。
- **適応二値化**: Otsu と Sauvola のフォールバックを使います。
- **画像品質の評価**: コントラスト、二値化、傾きをもとに処理を選びます。
- **レイアウト解析**: 射影、連結成分、縦書き検出に対応します。
- **ルビ分離**: 連結成分のサイズからふりがなを分離します。
- **カスケード認識**: 文字分割、分類、CTC フォールバックを組み合わせます。
- **純 Rust 推論**: `ocrus-nn` で PP-OCRv5 を実行します。ONNX Runtime は不要です。
- **`.ocnn` モデル**: mmap 読み込み、型付きグラフ、Conv+BN+ReLU 融合に対応します。
- **CTC デコード**: Greedy デコードを使い、信頼度が低い行には Beam Search を適用します。
- **入力への耐性**: 短い logits や NaN を含む入力もパニックせずに処理します。
- **日本語文字セット**: JIS X 0208 の文字セットで logits をマスクします。
- **辞書補正**: Aho-Corasick による後処理補正を行います。
- **メモリマップ I/O**: mmap で画像データを読み込みます。
- **並列正規化**: rayon でテキスト行を処理します。
- **出力形式**: JSON とプレーンテキストに対応します。
- **訓練データ**: フォントスタイルを絞り、拡張処理付きの文字画像を生成します。
- **ファインチューニング**: PP-OCRv5 と PaddleOCR の訓練手順に対応します。
- **対話型 TUI**: 文字認識に関連する操作をターミナルのメニューから実行できます。

CLI と Python バインディングは `ocrus-engine` の認識パイプラインを共有します。処理の流れと全9クレートは [アーキテクチャ](docs/architecture.ja.md) を参照してください。

## 動作環境

Linux と macOS 向けのリリースバイナリがあります。ソースからのビルドには Rust 1.99 以降が必要です。開発用ツールの版は `mise.toml` に固定しています。

## インストール

<!-- standard:install:start -->
### Cargo

Rust 1.99 以上が必要です。

```bash
cargo install --git https://github.com/owayo/ocrus ocrus-cli --locked
```

### GitHub Releases から

[Releases](https://github.com/owayo/ocrus/releases/latest) から自分の環境のアーカイブを取得して展開し、`ocrus` を `PATH` の通った場所に置きます。各リリースには、取得したファイルを確かめるための `SHA256SUMS` も添付しています。

| プラットフォーム | ファイル |
|---|---|
| Linux (x86_64) | `ocrus-x86_64-unknown-linux-gnu.tar.gz` |
| macOS (Intel) | `ocrus-x86_64-apple-darwin.tar.gz` |
| macOS (Apple Silicon) | `ocrus-aarch64-apple-darwin.tar.gz` |

macOS でブラウザから取得した場合は、実行の前に隔離属性を外します: `xattr -d com.apple.quarantine ocrus`。

### ソースから

[mise](https://mise.jdx.dev/) が必要です (Rust のツールチェーンは `mise.toml` で固定しています)。

```bash
git clone https://github.com/owayo/ocrus.git
cd ocrus
make install
```

`make install` は `/usr/local/bin` に入れます。場所を変えるときは `INSTALL_PATH` を指定します (例: `make install INSTALL_PATH="$HOME/.local/bin"`)。
<!-- standard:install:end -->

## クイックスタート

認識の前にモデルをダウンロードして変換します。モデルのセットアップはリポジトリのチェックアウトから実行してください。変換には数分かかる場合があります。

```bash
./models/download.sh
ocrus recognize image.png
ocrus recognize image.png --format json
ocrus tui
```

モデルの既定の場所は `~/.ocrus/models/` です。別の場所を使うには `OCRUS_MODEL_DIR` を設定します。ダウンロードと ONNX 変換は [モデルのセットアップ](docs/models.ja.md) を参照してください。

## 使い方

### 文字認識

```bash
ocrus recognize image.png --charset jis
ocrus recognize image.png --dict corrections.txt
ocrus recognize image.png --mode fastest
ocrus recognize image.png --mode accurate
ocrus recognize image.png --ruby
```

カスケード認識、TUI の操作、ベンチマーク、データセットのコマンドは [使い方の詳細](docs/usage.ja.md) に記載しています。ベンチマークの反復回数は1以上を指定します。

### Python

```python
import ocrus

engine = ocrus.OcrEngine()              # モデルを1度読み、複数の画像で使い回す
result = engine.recognize("page.png")   # パス、バイト列、numpy 配列、PIL Image
print(result.full_text())
for line in result.pages[0].lines:
    print(line.bbox.as_tuple(), round(line.confidence, 3), line.text)
```

`to_json()` は CLI の `--format json` と同じ形式です。numpy は任意です。[Releases](https://github.com/owayo/ocrus/releases/latest) から abi3 wheel を取得するか、`make wheel` でビルドできます。インストールと API の詳細は [Python パッケージ](python/README.md) を参照してください。

### 訓練とモデル

モデルは `.ocnn` バイナリ形式を使い、f16 の重みとゴールデン出力を保存します。詳細は [フォーマット仕様](docs/ocnn-format.ja.md) を参照してください。

データ生成、文字カテゴリ、フォントスタイル、PaddleOCR のファインチューニング、ONNX エクスポート、INT8 量子化、精度テストは [訓練](docs/training.ja.md) にまとめています。描画サイズ、一般的なデータ選定基準、評価の制約は [小さい文字の学習](docs/training-small-text.ja.md) に記載しています。訓練とモデル変換は Rust の推論エンジンとは別に実行します。

学習データと大規模な評価画像は同梱していません。手元で生成したデータを使って学習・評価します。

## 開発

<!-- standard:dev:start -->
[mise](https://mise.jdx.dev/) が必要です。ツールの版は `mise.toml` で固定しています。

```bash
make setup   # ツールチェーン (mise) と依存を取得する
make ci      # CI と同じ検査 (書き換えない)
```

| コマンド | 説明 |
|---|---|
| `make setup` | ツールチェーン (mise) と依存を取得する |
| `make build` | デバッグ版をビルドする |
| `make release` | リリース版をビルドする |
| `make run` | デバッグ版を実行する (引数は ARGS="...") |
| `make test` | テストを実行する |
| `make lint` | clippy を警告ゼロで通す |
| `make fmt` | コードを整形する (書き換える) |
| `make fmt-check` | 整形済みかを確かめる (書き換えない) |
| `make check` | 整形と静的検査 (書き換えない) |
| `make ci` | CI と同じ検査 (書き換えない) |
| `make install` | リリース版を INSTALL_PATH (既定 /usr/local/bin) に入れる |
| `make uninstall` | INSTALL_PATH から取り除く |
| `make clean` | ビルド成果物を消す |

`make` でターゲットの一覧を表示します。リリースは GitHub Actions で行います (**Actions → Release → Run workflow**)。
<!-- standard:dev:end -->

`make test` は macOS の Python ライブラリ探索パスを設定します。モデルがない場合はモデル依存のテストが省略されるため、単体テストの成功だけで OCR 精度は確認できません。追加のコマンド、Python wheel の検査、検証記録は [開発の詳細](docs/development.ja.md) を参照してください。

## ライセンス

<!-- standard:license:start -->
[MIT](LICENSE)
<!-- standard:license:end -->
