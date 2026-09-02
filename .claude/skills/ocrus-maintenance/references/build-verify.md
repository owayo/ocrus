# ビルドと検証の症状別対応

## コマンド一覧

```bash
cargo build --workspace
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --all -- --check
```

`--workspace` が要るのは、ルートに `ocrus-bench` パッケージがあるせいで、
付けないと cargo がルートパッケージしか見ないため。

OCR のスモーク（約 20 秒、要モデル）:

```bash
cargo test -p ocrus-engine --release --test smoke -- --nocapture
```

Python バインディング:

```bash
cd python && uvx maturin build --release --out ../target/wheels
```

```bash
uv run --with ./target/wheels/<wheel> --with pytest python -m pytest python/tests -q
```

CLI の手動確認:

```bash
cargo run --release -p ocrus-cli -- recognize testdata/sample_ja.png --format json
```

## 症状別

| 症状 | 原因 | 対応 |
| --- | --- | --- |
| `cargo test` が `ocrus-python` のリンクで落ちる（Linux / macOS） | `pyo3/extension-module` が有効になっている。この feature を付けると libpython にリンクしない | `crates/ocrus-python/Cargo.toml` の `default` に `extension-module` を入れない。有効化は maturin だけが行う（`python/pyproject.toml` の `features`） |
| `ImportError: cannot import name '_native'` | `python/` の中から pytest を回して、ソースの `python/ocrus/` が wheel を隠している | リポジトリのルートから `python -m pytest python/tests` |
| `ModelNotFoundError` / スモークが skip される | `~/.ocrus/models` に `rec.ocnn` と `dict.txt` が無い | `python models/download.py` で ONNX と辞書を取り、`scripts/src/ocrus_scripts/convert_to_ocnn.py` で `.ocnn` に変換する（長いのでユーザーに依頼） |
| スモークが `no fonts in test_images` で skip | 精度テスト用の画像が未生成 | `cargo test -p ocrus-cli --test generate_test_images -- --ignored --nocapture`（長いのでユーザーに依頼） |
| スモークの `non-empty` が 0 になった | レイアウトかデコードが死んでいる。依存更新なら `image` / `wide` / `ndarray` が怪しい | `git diff` で該当パッケージを確認し、1 つずつ戻して切り分ける |
| 単文字の正解率が 0% | 現状どおり。本番パイプラインは `normalize_line_scaled` と TLA をまだ使っていない | 異常ではない。SKILL.md 手順 3 の注記を参照 |
| 認識が debug ビルドで極端に遅い | 推論が純 Rust なので最適化なしでは 100 倍以上遅い | 必ず `--release` を付ける |
| `cargo clippy --workspace` が触っていない crate で警告 | `ocrus-nn` / `ocrus-preproc` / `ocrus-layout` に元からある 4 件 | 直さずに報告する。CI は `--workspace` を付けていないので落ちない |
| `cargo fmt -- --check` が差分を出す | 自分の編集が未整形、または元から未整形 | 自分が触ったファイルだけ `cargo fmt -p <crate>` |
| `maturin` が Python を見つけられない | 対象の interpreter が無い | `uvx maturin build -i python3.12` のように明示する。wheel は abi3-py310 なので 3.10 以上ならどれでもよい |
| wheel は作れたが `import ocrus` で落ちる | `[lib] name` と `module-name` の対応が崩れた | `crates/ocrus-python/Cargo.toml` の `#[pymodule] fn _native` と `python/pyproject.toml` の `module-name = "ocrus._native"` が一致していることを確認 |
| `cargo build` が `include_str!` で落ちる | `data/test_chars/*.txt` が消えた／移動した | `Charset::from_jis_embedded()` がコンパイル時に読んでいる。ファイルを戻す |
| CLI と Python で結果が違う | パイプラインが二重化している | 認識ロジックは `ocrus-engine` にしか置かない。CLI か Python 側にロジックが漏れていないか確認する |

## テストの構造

| テスト | 場所 | モデル | 時間 |
| --- | --- | --- | --- |
| 単体テスト | 各 crate の `src/` 内 | 不要 | 数秒 |
| 推論精度（数値の一致） | `crates/ocrus-nn/tests/precision.rs` | 不要 | 数秒 |
| OCR スモーク | `crates/ocrus-engine/tests/smoke.rs` | 要（無ければ self-skip） | 約 20 秒 |
| Python バインディング | `python/tests/test_ocrus.py` | 要（無ければ self-skip） | 約 30 秒 |
| 文字精度 | `crates/ocrus-cli/tests/char_accuracy.rs` | 要 | step1 約 36 分 / step2 約 3 分 |
| テスト画像生成 | `crates/ocrus-cli/tests/generate_test_images.rs` | 不要 | 長い |

`#[ignore]` が付いているのは最後の 2 つ。`cargo test` では動かない。
