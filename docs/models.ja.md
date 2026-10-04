# モデルのセットアップ

変換には Python 3.12 と uv が必要です。リポジトリのルートで `make setup` を実行して固定した版のツールを用意し、OCR モデルをダウンロードして `.ocnn` フォーマットに変換します。

```bash
./models/download.sh
```

デフォルトでは `~/.ocrus/models/` にインストールされます。`OCRUS_MODEL_DIR` 環境変数で変更可能です。

- `rec.ocnn` — PP-OCRv5 認識モデル（純 Rust 推論、ONNX Runtime 不要）
- `dict.txt` — 文字辞書（18,383 文字）

ONNX モデルを `.ocnn` フォーマットに変換するには：

```bash
uv run --project scripts --extra convert python scripts/src/ocrus_scripts/convert_to_ocnn.py rec.onnx -o ~/.ocrus/models/rec.ocnn
```

## `.ocnn` フォーマット

`.ocnn`（**Oc**rus **N**eural **N**etwork）は OCRus の純 Rust 推論エンジン（`ocrus-nn`）向けに設計されたカスタムバイナリモデルフォーマットです。ONNX Runtime への依存を排除しつつ、`mmap` によるゼロコピーモデルロードを実現します。

主な特徴：
- **mmap 対応**：テンソル本体をマップし、JSON メタデータを解析・検証
- **Conv+BN+ReLU 融合**：変換時にバッチ正規化を畳み込みに融合
- **型付きグラフ**：名前付きパラメータ、SSA の値参照、動的次元式
- **f16 重み**：既定のモデルは約40MB。半精度の全ビットパターンを単体テストで検証
- **ゴールデン出力**：変換時の出力と実行結果を比較可能

詳細は [モデルフォーマット](ocnn-format.ja.md) を参照してください。
