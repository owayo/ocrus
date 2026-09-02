# ocrus (Python)

Pure Rust の日本語 OCR エンジン [ocrus](https://github.com/owayo/ocrus) を Python から使うためのバインディング。

推論は Rust 側で完結する。ONNX Runtime も PaddlePaddle も要らず、numpy すら必須ではない。

```python
import ocrus

engine = ocrus.OcrEngine()  # モデルを 1 度だけ読む
result = engine.recognize("page.png")

print(result.full_text())
for line in result.pages[0].lines:
    print(line.bbox.as_tuple(), line.confidence, line.text)
```

## 入力

`recognize()` は 4 種類を受け取る。

| 渡すもの | 例 |
| --- | --- |
| パス | `engine.recognize("page.png")` |
| エンコード済みバイト列 | `engine.recognize(Path("page.png").read_bytes())` |
| numpy 配列 (uint8, `(H,W)` / `(H,W,3)` / `(H,W,4)`) | `engine.recognize(np_image)` |
| PIL の Image | `engine.recognize(Image.open("page.png"))` |

numpy と PIL は任意。どちらも入っていなければ、パスとバイト列だけで使える。

## オプション

```python
engine = ocrus.OcrEngine(
    model_dir="/path/to/models",  # 既定: $OCRUS_MODEL_DIR か ~/.ocrus/models
    mode="accurate",  # auto / fastest / accurate
    charset="jis",  # full / jis（JIS 外の文字を出さない）
    dict_path="corrections.txt",  # 後処理の補正辞書
    ruby=True,  # ルビを本文から分離する
)
```

## モデル

`rec.ocnn` と `dict.txt` がモデルディレクトリに要る。無い場合は `ModelNotFoundError` が上がる。

```python
import ocrus

ocrus.models_ready()  # True / False
ocrus.default_model_dir()  # 探しに行く場所
```

モデルの用意はリポジトリ側の手順に従う（`python models/download.py` で ONNX と辞書を取得し、
`scripts/src/ocrus_scripts/convert_to_ocnn.py` で `.ocnn` に変換する）。
変換済みの `rec.ocnn` を配れる場所があるなら、そこへ置いて `OCRUS_MODEL_DIR` を向ければよい。

## 例外

| 例外 | いつ |
| --- | --- |
| `ocrus.ModelNotFoundError` | モデルディレクトリに `rec.ocnn` / `dict.txt` が無い |
| `ocrus.ModelError` | モデルはあるが読めない・推論に失敗した |
| `ocrus.ImageError` | 画像を読めない・デコードできない・生ピクセルの形が合わない |
| `ocrus.ConfigError` | `mode` / `charset` の値が不正、補正辞書を読めない |

すべて `ocrus.OcrusError` を継承しているので、まとめて捕まえられる。

## スレッド

認識中は GIL を解放する。1 つの `OcrEngine` を複数スレッドから呼べば実際に並列に走るので、
エンジンをスレッドごとに作り直す必要はない（作り直すと 80MB のモデルを何度も読むことになる）。

## ビルド

```bash
cd python
uvx maturin develop --release                      # 開発用（今の venv に入れる）
uvx maturin build --release --out ../target/wheels # wheel を作る
```

テストはリポジトリのルートから回す。`python/` の中から回すと、ソースの `python/ocrus` が
インストール済みの `ocrus` を隠して `_native が無い` で落ちる。

```bash
python -m pytest python/tests -q
```
