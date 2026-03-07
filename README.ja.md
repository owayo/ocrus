# ocrus

世界最速を目指す日本語特化OCR (Rust実装)

## 機能

- 横書き日本語印刷文字認識
- ONNX Runtime推論バックエンド
- 水平投影プロファイルによる行検出
- Otsu二値化前処理
- CTC Greedyデコーディング
- JSON / プレーンテキスト出力

## 動作環境

- Rust 2024 edition (1.85+)
- macOS / Linux

## インストール

```bash
cargo install --path crates/ocrus-cli
```

## 使い方

```bash
# 画像からテキスト認識
ocrus recognize image.png

# JSON出力
ocrus recognize image.png --format json

# モデルディレクトリ指定
ocrus recognize image.png --model-dir /path/to/models

# ベンチマーク
ocrus bench image.png -n 100
```

## モデルのセットアップ

```bash
./models/download.sh
```

デフォルトのモデルディレクトリ: `~/.ocrus/models/`
環境変数 `OCRUS_MODEL_DIR` でオーバーライド可能。

## 開発

```bash
cargo build          # ビルド
cargo test           # テスト
cargo clippy         # リント
make check           # clippy + check
make bench           # ベンチマーク
```

## ライセンス

MIT
