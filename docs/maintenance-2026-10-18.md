# 2026-10-18 保守記録

## 同期と依存更新

`main` を `origin/main` と同期し、開始時点は `f6145aa6`。未コミット変更はなかった。
Rust 1.99.0、Python 3.12.14、maturin 1.15.0 は維持した。

| 対象 | 指定の更新 |
| --- | --- |
| rand | 0.10.2 → 0.10.3 |
| serde_json | 1.0.151 → 1.0.152 |
| thiserror | 2.0.20 → 2.0.21 |
| pyo3 | 0.29.2 → 0.29.3 |
| tempfile（テスト用） | 3 → 3.27.0 |
| ONNX Runtime（変換・量子化用） | 1.30.0 → 1.31.0 |
| uv / Ruff | 0.12.23 → 0.13.0 / 0.16.7 → 0.17.0 |

Rust の一部は既にロック済みの版に指定下限を揃える更新で、上表のすべてが実際の
解決バージョンの変更ではない。Cargo.lock の間接依存も更新した。
`image`、`imageproc`、`ndarray`、`wide` の版は変更していない。

`depup --dry-run --include-pinned` の後、
`depup --install --include-pinned --exclude python --exclude numpy --exclude onnxruntime`
を実行した。Python 3.12 の訓練用制約と、利用者向けの `numpy>=1.24` は維持した。
重複宣言を depup が更新できなかった ONNX Runtime は、2か所の指定と uv.lock を個別に更新した。

`pest` 系 2.9.3 は、新たに C コンパイラを使う `psm` / `stacker` をロックファイルへ
追加するため、`pest` / `pest_derive` / `pest_generator` / `pest_meta` を 2.9.2 に戻した。
追加された2依存も削除されている。メジャー更新は採用していない。

ONNX Runtime の推論だけでは量子化モジュールの依存不足を検出できなかった。
`quantize` extra を `onnxruntime[quantization]` と `onnx` の組み合わせにし、
`ml-dtypes` が解決されるようにした。量子化 extra の定義は
[ONNX Runtime 1.31.0 の配布設定](https://github.com/microsoft/onnxruntime/blob/v1.31.0/setup.py)
でも確認した。

## 確認できた不具合と修正

| 場所 | 再現条件と影響 | 修正 |
| --- | --- | --- |
| 事前描画画像からのデータ生成 | `--samples-per-char` を参照しておらず、0指定でも画像を生成。3回指定でも各拡張を1回しか出力しない | 各拡張を指定回数生成し、進捗表示にも回数を反映 |
| 失敗例からの生成の引数 | `--all-fonts --test-images` が受理されるが、前者がフォント名を消すため全画像がスキップされる | clap で併用を拒否し、それぞれ単独では受理 |
| 既定のモデル探索 | Windows で `HOME` が未設定だと、配置済みモデルを見つけず作業ディレクトリ内を探索 | `HOME` がなければ `USERPROFILE` を使用。TUI も共通のパス解決を使用 |

3件とも修正前に失敗する回帰テストを実行し、修正後の成功を確認した。
環境変数の優先順位は、プロセスの環境を変更しない注入可能な読み取り関数で検証する。

データ生成では0・1・3回指定、重複した失敗例、欠損画像、ラベル数、PNG数、
元画像の画素と寸法を確認する。実際の CLI でも、合成画像1枚・拡張5種・3回指定から
15枚を生成し、相反する入力元の指定が終了コード2になることを確認した。
量子化には、最小構成の環境で MatMul を INT8 に変換し、変換後の演算と推論結果を
確認する Python テストを1件追加した。Rust の追加テストは3件。

## 構造とレビュー

開始時のソース90ファイルを `astro-sight symbols` で確認し、入力境界、モデルの範囲・形状、
デコード、並列データ生成、ファイル出力、外部コマンドの引数を重点的に調べた。
変更後は `context` / `impact` / `review --git` で参照先と差分を確認した。
公開 API の追加・削除・シグネチャ変更、未対応の影響、欠けた関連変更は検出されなかった。
既存の公開補助 API は、リポジトリ内での参照がないだけでは不要と断定しない。

複雑度の上位には演算の dispatch や文字対応表が含まれるため、数値だけを根拠に分割しない。
TUI の重複したモデルパス解決の集約以外に、リファクタリングが必要な根拠はなかった。
認識処理を `ocrus-engine` に集約する構造は維持している。

## 検証結果

Windows 上で mise の固定ツールを使用した。GNU Make がなかったため、`mise install` と
`cargo fetch --locked` の後、Makefile と同じ Cargo コマンドを直接実行した。

| 検査 | 結果 |
| --- | --- |
| `cargo build --locked --workspace` | 成功 |
| `cargo test --locked --workspace` | 233件成功、失敗0、ignore指定10件。モデル依存の実行は下記 release 検査で別途確認 |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | 成功、警告0 |
| `cargo fmt --all -- --check` / `cargo check --locked --workspace` | 成功 |
| `cargo build --release --locked -p ocrus-cli` | 成功 |
| Python wheel / numpy 1系・2系での pytest | abi3 wheel のビルド成功。numpy 1.26.4 と 2.5.4 のそれぞれで16件成功。numpy / Pillow なしの基本インストールでも実画像を認識 |
| `ruff check python scripts/tests` / `ruff format --check python scripts` | 成功 |
| `ruff check scripts` | 既存の指摘5件。Ruff 0.16.7 と 0.17.0 の両方で同じ箇所・規則を確認 |
| `uv lock --project scripts --check` | 成功 |
| 量子化 extra だけの分離環境での unittest | 1件成功。INT8 の MatMul と推論値を確認 |
| ONNX Runtime 1.31.0 での実モデル推論 | 成功。入力 `(1,3,48,104)`、出力 `(1,13,18385)`、全値が有限 |
| CLI の事前画像生成・引数排他・既定モデル探索 | 成功 |
| release の OCR スモーク | 2行で「こんにちは世界 / 日本語OCRテスト」。単文字17/24正解・24/24非空。変更前と同じ集計 |
| モデルのゴールデン検証 | チェックサムと2/2レコードの一致を確認 |
| 本番パイプラインの文字精度（通常モード・全文字集合） | ひらがな385/415、カタカナ359/430、合計744/845（88.0%）。空出力3件、実行50.2秒 |

既存の Ruff 指摘は、外部コマンドの `check` 明示2件、不要な noqa 1件、
例外フォールバックの2規則。今回の更新による新規の指摘ではない。

本番の文字精度テストには ignore 指定があるため、実測は次のコマンドで別途実行した。
既存の評価結果を退避し、新しい結果を別名で保存してから既存ファイルを復元した。

```bash
cargo test --locked -p ocrus-engine --release --test accuracy \
  production_pipeline_character_accuracy -- --ignored --nocapture
```

## モデル形式の計測と判断

同じ f16 のモデルで、ロード11回の中央値と、2回のウォームアップ後の W=104 の
ゼロ入力5回の中央値を測った。op の内訳は `Executor::run_profiled()` の5回平均。

| 指標 | 実測 |
| --- | --- |
| ファイルサイズ | 42,101,410 bytes（40.2 MiB） |
| ロード | 0.746 ms |
| 推論（W=104） | 69.343 ms |
| Conv | 57.549 ms、op 合計の84.99% |
| Binary / Concat / MatMul / Softmax | 6.29% / 2.79% / 2.93% / 1.46% |

過去の62 ms・Conv約83%とは同じ測定条件を保証していないため、速度の回帰判定には使わない。
サイズとボトルネックの傾向は従来と同じで、ロードは1 ms未満。
今回は形式変更を行う根拠がなく、int8 化・重み配置変更・CBOR 化は見送った。

## 文書と未実施の範囲

AGENTS.md に PowerShell での検査とモデル探索、README と学習文書に事前画像の生成回数・
入力元の排他条件・量子化用依存とテスト手順を反映した。README は日本語のため
README.ja.md は追加していない。

指定された `link_agent.md.sh` はこの Windows 環境に存在せず、同等のスクリプトも
確認できなかったため未実行。CLAUDE.md は Git 上ではシンボリックリンクだが、この
チェックアウトでは9バイトの通常ファイルとして展開されている。直接編集・置き換えは行わない。

認識アルゴリズムや画像・数値処理の依存は変更しておらず、長時間の独自経路の
`char_accuracy` step1 / step2 は未実行。変更前後の正式な精度比較は行っていない。
GPU 訓練と Linux / macOS 上での検証も未実施。
学習・評価データ、モデル、具体的なフォント名、ローカルの実行ログはコミットしない。
