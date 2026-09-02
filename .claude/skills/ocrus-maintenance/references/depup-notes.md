# depup の落とし穴（ocrus 版）

## depup が見る範囲

`.depup` に `scripts` と `python` が書いてある。実行するとこれだけの単位を別々に見る。

| 単位 | ファイル | 中身 |
| --- | --- | --- |
| ルート workspace | `Cargo.toml` | `[workspace.dependencies]` の共通指定 |
| 各 crate | `crates/*/Cargo.toml` | workspace を経由しない直書きの依存 |
| 学習・変換スクリプト | `scripts/pyproject.toml` | PaddlePaddle 系。学習と ONNX 変換でしか使わない |
| Python バインディング | `python/pyproject.toml` | maturin のビルド設定と numpy の任意依存 |

crate ごとに直書きされている依存（`wide`、`memmap2`、`imageproc` など）は
workspace 側を直しても上がらない。`depup --dry-run` の出力は crate 単位で読むこと。

## 上げ方に注意が要るもの

### `wide`（SIMD の土台）

`ocrus-preproc` / `ocrus-layout` / `ocrus-nn` が直に使っている。グレースケール化・二値化・
正規化・射影・畳み込みの内側にいるので、**ここが変わると認識結果が変わりうる**。
上げた回は必ず手順 3 のスモークを回し、OCR に効く更新として手順 4 の精度依頼も検討する。
バージョン番号が minor でも、`f32x8` 系の演算の丸めが変われば出力は動く。

### `image` / `imageproc`

デコードとリサイズの実装が変わると入力画素そのものが変わる。patch 更新でも
OCR に効く更新として扱う。`imageproc` は `ocrus-layout` の CCL と `ocrus-dataset` の
フォント描画の両方で使っているので、精度テスト用の画像生成にも影響する
（`test_images/` を作り直すと、以前のログと分母が合わなくなる点に注意）。

### `daachorse`（辞書補正の Aho-Corasick）

1.0 → 5.0 のようなメジャー更新を提案してくる。単独で当てて、`--dict` を渡す経路
（`ocrus-recognizer` の `DictCorrector`）が動くことを確かめてから次へ進む。
辞書補正は既定では無効なので、テストが緑でも動作確認にならない点に注意。

### `serde_yaml`

`0.9.34+deprecated` が最終版。作者が開発をやめた crate なので、これ以上は上がらないし
上げようとしなくてよい。`ocrus-cli` の dev-dependency（`char_accuracy` の設定読み込み）。

### `ratatui`

TUI (`ocrus tui`) だけで使う。メジャー更新はウィジェット API が毎回変わるので、
上げた回は `cargo build` が通るだけでなく、TUI を一度起動して確認する価値がある。
ここが壊れても OCR の精度には影響しない。

## Python 側

### `scripts/pyproject.toml`（学習・変換）

- `requires-python = ">=3.12,<3.13"` は **PaddlePaddle が 3.13 に対応していない**ため。
  上限を外さない。
- `[tool.uv]` の `extra-index-url`（PaddlePaddle の CUDA ビルド）と
  `index-strategy = "unsafe-best-match"` は消さない。消すと paddlepaddle-gpu が解決できない。
- `uv sync --extra train` は数 GB のダウンロードになる。AI セッションからは実行しない。
- `onnxruntime` は量子化 (`quantize`) 用、`paddlex` は学習用。どちらも通常の OCR 実行には
  要らないので、更新しても手順 2〜3 の結果は変わらない。

### `python/pyproject.toml`（バインディング）

- `numpy = ["numpy>=1.24"]` は**任意 extra の下限**であって、使うバージョンの指定ではない。
  depup はここを最新版まで引き上げようとするが、**上げない**。上げると numpy 1.x を使っている
  利用者が理由もなく弾かれる。バインディング自体は numpy に依存していない
  （配列は buffer protocol 経由で読む）。
- `maturin>=1.9,<2.0` はビルドバックエンド。メジャーを跨ぐときは wheel が問題なく作れるかを
  実際に確かめてから。
- `crates/ocrus-python/Cargo.toml` の `pyo3` を上げたら、`abi3-py310` feature が生きているかと
  `python -m pytest python/tests` が通るかを両方見る。pyo3 のメジャー更新は
  `#[pyclass]` 周りの非推奨警告としてまず現れる。

## 越えてはいけない線

このプロジェクトの売りは「Pure Rust・外部ランタイム依存ゼロ・C/C++ ツールチェーン不要」。

**ONNX Runtime、OpenCV、cmake や C/C++ コンパイラを必要とする crate を持ち込まない。**
依存更新でそうしたものが入ってきたら、更新を当てずに報告して止まる。
ビルドが通るかどうかの問題ではなく、プロジェクトの前提の問題なので、判断はユーザーが行う。
