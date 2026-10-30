# 2026-10-30 保守記録

## 同期と依存更新

開始時の HEAD は `64e046ea`。未コミット変更、未送信コミットはなく、`main` を
`origin/main` と同期した（既に最新）。stash は使用していない。

`mise install` と変更前の workspace 検証（build / test / clippy / fmt）を通した後、
`depup --dry-run` と `depup --install --exclude python --exclude numpy` を実行した。

| 対象 | 更新 |
| --- | --- |
| uv | 0.13.4 → 0.13.5 |
| ruff | 0.17.1 → 0.17.2 |

Rust の直接依存、PyO3、画像処理・SIMD の依存、Python パッケージには更新がなかった。
`depup --dry-run --include-pinned` でも追加の候補は出ず、学習用の Python 3.12 と
任意 extra の `numpy>=1.24` は据え置いた。前者は `scripts/pyproject.toml` の
`requires-python = ">=3.12,<3.13"`（PaddlePaddle の制約）に合わせるため、後者は
任意 extra の下限を上げると numpy 1.x の利用者を弾くため。この 2 つの除外指定を
保守スキルの手順へ反映した。

`pest` 系の推移依存更新は、新たに C コンパイラを必要とする `psm` / `stacker` を
追加するため見送った（前回と同じ判断）。`cc` / `rustix` / `smallvec` /
`unicode-width` / `uuid` の更新候補も、脆弱性の解消を伴わないため Cargo.lock を
据え置いた。`ocrus-recognizer` の `ndarray` 依存は下の削除に伴って外れた。

## GitHub Issue

開始時の open Issue は 0 件。点検中に見つけた不具合を 2 件起票し、同じ点検で修正した。

| Issue | 内容 | 対応 |
| --- | --- | --- |
| [#3](https://github.com/owayo/ocrus/issues/3) | `--cascade` を指定してもカスケード認識が実行されない | 未実装の経路とドキュメントを削除 |
| [#4](https://github.com/owayo/ocrus/issues/4) | `--mode fastest` が auto と同じ挙動で、品質ゲートの計算結果が使われない | 未使用の計算を削除し、コメントとドキュメントを実態へ合わせた |

## 効かない設定の削除

`--cascade` は `EngineConfigBuilder::cascade_model_path` へ渡るが、`ocrus-engine` は
`cascade_model_path` も `cascade_threshold` も読んでいなかった。`CascadeRecognizer` 自体も
分類器の推論が未実装のプレースホルダで、入力に関係なく空文字列と `used_ctc: true` を返す。
`ocrus-engine` からの参照もない。指定しても何も起こらない設定とオプション、
プレースホルダ実装、`cascade` から呼ばれるだけだった `segment_characters` を削除し、
`AGENTS.md`・`README.md`・`docs/usage*.md`・`docs/architecture*.md` の記述も外した。

`prepare_image` では `assess_quality` と `should_use_fast_path` の戻り値が
`_use_fast_path` に束縛されて捨てられていた。実際の分岐は
`matches!(config.mode, OcrMode::Accurate)`（前処理バリアントの多数決）だけで、
`Fastest` と `Auto` を区別する箇所がない。結果を使わない計算を外し、
「`OcrMode` はパイプラインを変えない」という実態と合わないコメントを書き直した。
品質評価は `ocrus-layout` の API として残るため、`GlyphCache` と同じ扱いで
「engine は使わない」と明記した。`--mode fastest` に別の意味を与えるか落とすかは
Issue #4 に残した。

## 構造の整理

`ctc_greedy_decode` と `ctc_greedy_decode_masked` は、マスク適用の 3 行を除いて
入力検証・argmax・信頼度の集計・畳み込みが同一だった。前回の保守で
「マスク長検査と異常入力時の挙動を別途保持する必要がある」として見送った課題で、
今回は検証を `validated_frames` に切り出し、argmax をスコア変換のクロージャを取る
`decode_argmax` に統合した。マスク長の `assert_eq!` は入力検証の後という位置を保ち、
短い入力ではマスク長を見ないことを回帰テストで固定した。`ctc_tla` と `ctc_beam` も
同じ検証を使い、時刻ごとの走査を `chunks_exact` に置き換えた。
`ctc_greedy.rs` は 239 行から 194 行になり、重複していた検証は 4 か所から 1 か所になった。

参照のない公開シンボル `Quad`、`available_categories`、`render_single_char` と、
cascade に伴う `EngineConfigBuilder::cascade_model_path` / `cascade_threshold` を削除した。
前回は「公開 API であるため」残したが、cascade は機能自体が未接続で、残る 3 件も
定義以外の参照が無く、`astro-sight dead-code` の報告が空になった。
`astro-sight review --git` の `api_changes.removed` / `.modified` は空で、
削除分はすべて `removed_dead` に入っている。

`generate_test_chars.py` の常用漢字一覧の取得は `except Exception: pass` で
失敗を捨てていた。取得と解析のどこで失敗したか分からないため、理由を stderr へ出し、
既存ファイルへの退避はそのまま残した。

## 性能の測定条件

Windows x86_64、Ryzen 7 5700X（8 コア・16 スレッド）、Rust 1.99.1 の release、
LTO 有効・codegen-units=1 で測定した。比較用バイナリを保存し、
前→後、後→前、前→後の順で 3 組測る。criterion と `maintenance_profile` は
論理 CPU 14 に割り当て、優先度を AboveNormal、`RAYON_NUM_THREADS=1` に固定した。
本番パイプラインと `recognize` の計測は既定の並列設定で、優先度のみ固定している。

各ケースは 1 秒ウォームアップ後、40 サンプルを約 3 秒かけて収集する。
生サンプルと 95% 信頼区間は [測定データ](benchmarks/maintenance-2026-10-30.json) に保存した。

```bash
mise exec -- cargo bench --locked --bench preproc -- "ctc_greedy|ctc_beam|quality_assess" --sample-size 40 --warm-up-time 1 --measurement-time 3 --noplot --save-baseline <tag>
```

## 性能の結果

| ベンチ | 変更前 中央値（最小〜最大） | 変更後 中央値（最小〜最大） | 差 |
| --- | ---: | ---: | ---: |
| ctc_greedy_100ts_101cls | 8.242 µs（8.204〜8.260） | 8.985 µs（8.767〜9.037） | +9.0% |
| ctc_beam_100ts_101cls_w5 | 11.331 ms（11.079〜11.338） | 11.177 ms（10.903〜11.737） | −1.4% |
| quality_assess_600x800（対照） | 10.534 ms（10.449〜10.874） | 11.672 ms（11.549〜11.683） | +10.8% |

**この条件では CTC の統合による性能差を判定できない。** 実装を変更していない
`quality_assess_600x800` が同じ測定で +10.8% 動き、範囲も重ならない。
バイナリ配置の違いによる系統差が ctc の差（±9%）より大きいため、
`ctc_greedy` の +9.0% も `ctc_beam` の −1.4% も変更の効果とは言えない。
統合の目的は重複の削除であり、性能改善としては主張しない。

本番パイプライン（単文字 845 枚、既定の並列設定）の所要時間は、
変更前が 42.6 / 42.8 / 43.2 秒（中央値 42.8）、変更後が 43.3 / 42.1 / 44.3 秒（中央値 43.3）。
正解数は両側すべて 744/845 で一致した。評価画像が小さく、削除した品質評価の寄与は
この幅に収まらない。

`testdata/large.png`（2400x800）の `ocrus recognize` 1 回は、変更前が
3149.3 / 2694.5 / 2859.7 ms（中央値 2859.7）、変更後が 2713.5 / 2655.0 / 2822.7 ms
（中央値 2713.5）。標準出力は一致した。変更前の範囲が 455 ms あり、
削除した品質評価の寄与（この画像サイズで 40 ms 程度）はばらつきを超えない。
削除の根拠は「結果を使わない計算の除去」であり、短縮量は確認できていない。

## モデル形式と推論

同じ f16 モデルは 42,101,410 bytes（40.2 MiB）。上記の CPU・スレッド条件で、
ロード 11 回、各幅の推論 5 回を計測した。計数用アロケータの負荷を含む。

| 指標 | 中央値（最小〜最大） |
| --- | ---: |
| ロード | 0.851 ms（0.827〜1.054） |
| W=104 推論 | 237.3 ms（233.4〜240.1） |
| W=320 推論 | 675.9 ms（669.6〜699.5） |

出力ハッシュは各幅の全試行で一意。op 合計に対する conv2d の割合は W=104 で 95.5%、
W=320 で 95.1%（次は matmul が 2.0% / 1.8%）。推論本体とモデル形式は変更していないため、
前回の保守記録の数値と直接比較しない（測定時のマシン状態の差を含む）。

[フォーマットの見直し項目](ocnn-format.md)は今回も採否を変えなかった。
ロードは推論の 0.4% 未満で CBOR 化の動機がなく、重みの重複排除は効果が小さい。
conv2d が 95% を占める状況は変わらないため、カーネル向けの事前並べ替えが
最も効く候補のままだが、conv の実装変更・変換器の対応・ゴールデンの再生成を伴うため
今回は着手しない。int8 化も `wide` の int8 内積と精度検証費用の検討が必要で据え置いた。

## 検証方法

| 検査 | 結果 |
| --- | --- |
| `cargo build --locked --workspace` | 成功 |
| `cargo test --locked --workspace` | 成功（recognizer は cascade/segment の 6 件が減り ctc の 5 件が増えて 38 件） |
| `cargo clippy --locked --workspace --all-targets -- -D warnings` | 警告なし |
| `cargo fmt --all -- --check` | 差分なし |
| `maturin build --release --locked` + `pytest python/tests` | 10 passed, 6 skipped（numpy 未導入分） |
| `cargo test -p ocrus-nn --release --test ocnn_golden` | ゴールデン 2/2 一致 |
| OCR スモーク | `sample_ja.png` 2 行、単文字 17/24 正解・24/24 非空（変更前と一致） |
| `char_accuracy_step1` | 662/1095（60.5%）、前回ログと全カテゴリ一致 |
| `char_accuracy_step2` | 613/845（72.5%）、前回ログと全カテゴリ一致 |
| 本番パイプライン精度 | 744/845（88.0%）、hiragana 385/415・katakana 359/430、変更前と一致 |
| `ruff check .` | 6 件 → 5 件 |

精度測定は保守スキルで解決したデータ置き場を `OCRUS_DATA_DIR` に設定して実行した。
フォント名、データの管理先、モデルや個別の評価結果は公開しない。

## 残課題

- `--mode fastest` に別の意味を与えるか落とすかは未決（Issue #4）。現状は `auto` と同じ経路。
- conv2d のカーネル向けレイアウトは効果が最も大きい候補のまま未着手（`docs/ocnn-format.md`）。
- `ruff check` の残り 5 件は実害がないと判断して直していない。`PLW1510` の 2 件は
  `subprocess.run` の戻り値を明示的に見て `sys.exit` しており、`RUF100` の 2 件は
  このリポジトリに ruff 設定がなく `S310` を有効にした環境では `noqa` が必要、
  `BLE001` の 1 件は取得と解析のあらゆる失敗で退避する意図的な捕捉。
  ruff は CI の検査に入っていない。
- GPU を使う学習、漢字全件の評価、他 OS でのローカル実行は未実施。
- 既定の並列設定での推論速度の退行は未判定（単一スレッド条件のみ測定）。
- `CLAUDE.md` は `AGENTS.md` へのシンボリックリンクではなく、`AGENTS.md` と書かれた
  通常ファイルのまま。保守スキルが参照する mac 用のリンク作成スクリプトは
  この環境に存在せず、同等のものも見つからないため未実行。
