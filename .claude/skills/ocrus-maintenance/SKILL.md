---
name: ocrus-maintenance
description: |
  ocrus（Pure Rust の日本語 OCR）を保守する。depup --install で Rust と Python の依存を
  更新し、cargo build / test / clippy / fmt を通し、Python バインディングのビルドとテストを
  確かめ、短い OCR スモークで認識が壊れていないことを見て、検証が通った変更だけをコミット
  するまでを 1 回の作業として扱う。
  「メンテナンスして」「依存を更新して」「depup を回して」「ビルドが通るようにして」
  「久しぶりに触るので整えて」「アップデートして問題ないか確認して」「定期メンテ」
  のような依頼では必ずこのスキルを使うこと。
  精度そのものを上げる作業（char_accuracy の結果解析、認識手法の実験、失敗文字の分析）は
  ocrus-model-improvement を使う。スモークが劣化していたらそちらへ引き継ぐ。
---

# ocrus のメンテナンス

このリポジトリは「依存を上げると壊れる場所」と「壊れても気づきにくい場所」が別々にある。
ビルドと clippy は前者を捕まえるが、**認識精度の劣化は cargo test では捕まらない**。
このスキルは、上流（依存）から下流（認識結果）へ順に確かめて、途中で落ちたら
そこが原因の層だと分かる 1 本道にする。

## このリポジトリの形

```
ocrus-core        型・EngineConfig・エラー（依存なし）
  ↑
ocrus-preproc / ocrus-layout / ocrus-recognizer / ocrus-nn    アルゴリズム層
  ↑
ocrus-engine      パイプライン本体（前処理 → レイアウト → 推論 → デコード）
  ↑                                    ↑
ocrus-cli         ocrus-python + python/ocrus（PyO3 / maturin）
```

**認識のロジックは ocrus-engine にしかない。** CLI も Python も薄い皮で、同じ結果を返す。
「CLI では直ったが Python では直っていない」が起きない形になっているので、ここを崩さないこと。

精度テスト `crates/ocrus-cli/tests/char_accuracy.rs` は本番と同じ正規化
（`normalize_line(g,b)` は `normalize_line_scaled(g,b,1.0)` と同一）を通り、
デコードだけ greedy と TLA を併用する。ほぼ本番と同じ経路だと思ってよい。

## AI が実行するもの / ユーザーに渡すもの

`AGENTS.md` のルール: **長時間かかるコマンドは AI 側で実行しない。** AI セッションが終わると
コマンドも死ぬので、30 分のテストを走らせても結果が残らない。

| AI が実行する（数秒〜数分） | ユーザーに依頼する（10 分〜） |
| --- | --- |
| `depup --dry-run` / `depup --install` | `char_accuracy` の各 step（step1 は約 36 分） |
| `cargo build` / `test` / `clippy` / `fmt` | `generate_test_images`（全フォント分の画像生成） |
| `cargo test -p ocrus-engine --test smoke`（約 20 秒） | ONNX → .ocnn 変換、モデルのダウンロード |
| `maturin build` と `pytest python/tests` | ファインチューニング、量子化、`uv sync --extra train` |
| ログと失敗リストの解析 | |

依頼するときはコマンドをそのまま渡し、**生成されたログのパスを返してもらう**。
渡しっぱなしにせず、返ってきたログを手順 4 の比較にかけるところまでが 1 回の作業。

## 0. 現状を掴む

```bash
python .claude/skills/ocrus-maintenance/scripts/repo_status.py
```

未コミットの変更、モデルの有無、release ビルドの鮮度、直近の精度、道具の有無が 1 画面で出る。

**未コミットの変更が残っていたら、depup を回す前にそれを独立したコミットとして確定させる。**
depup は `Cargo.toml` / `pyproject.toml` を書き換えるので、既存の変更と混ざると
「依存更新で壊れたのか元から壊れていたのか」が判別できなくなり、revert での切り分けもできない。

確認は取らずにコミットしてよい。ただし**壊れたものを確定させない**ため、コミットの前に
検証を通す。つまり未コミットの変更があった回は、**手順 2 → 3 を先に一巡してから手順 1 の
depup に戻る**（それ自体が「元から壊れていなかった」ことの確認にもなる）。
検証が通らなければコミットせず、そこで報告して止まる。stash は使わない — 戻し忘れると、
直したはずの変更が消えたように見える。

`logs/*.log`、`test_results/failures_*.json`、`test_images/`、モデルファイルはコミットしない。
これらは測定のたびに変わる生成物で、履歴に入れる価値がない。

## 1. 依存を更新する

いきなり `--install` せず、まず何が上がるかを見る。メジャー更新が混ざっていたら、
それだけ切り出して判断できるようにするため。

```bash
depup --dry-run
```

```bash
depup --install
```

リポジトリ直下の `.depup` に `scripts` と `python` が書いてあるので、depup は
**各 crate の Cargo.toml・`scripts/pyproject.toml`・`python/pyproject.toml`** をそれぞれ
別の単位として見る。crate に直書きされた依存（`wide`、`memmap2` など）は workspace 側を
直しても上がらないので、dry-run の出力は crate 単位で読むこと。

更新後、実際に何が変わったかを確認する。depup がどちらかの生態系を素通りしていることがある。

```bash
git diff --stat Cargo.toml Cargo.lock scripts/pyproject.toml scripts/uv.lock python/pyproject.toml
```

片方だけ更新されていないなら、明示的に回す。

```bash
depup --rust --install .
depup --python --install scripts
```

制約に手を出す前に [references/depup-notes.md](references/depup-notes.md) を読むこと。
このリポジトリには「上げてはいけない」理由が付いた依存がある。

## 2. ビルドと検査

CI (`.github/workflows/ci.yml`) と同じ 4 つを通す。ここが CI の合格条件そのもの。

```bash
cargo build --workspace
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --all -- --check
```

`--workspace` を付けること。ルートに `ocrus-bench` パッケージがあるため、付けないと
cargo はそれだけを見て、crate のテストが 1 つも走らない（CI もそうなっていた）。
`clippy` に `--workspace` を付けないのは CI と揃えるため。付けると
`ocrus-nn` / `ocrus-preproc` / `ocrus-layout` に元からある警告で落ちる。

`cargo test` にモデルは要らない。モデルが無い環境では OCR を実行するテストは自分で skip する
（`crates/ocrus-engine/tests/smoke.rs` の `engine_or_skip`）。CI が緑なのはそのため。

Python バインディングも触った回、あるいは `ocrus-engine` / `ocrus-core` の公開 API が
変わった回は、こちらも通す。

```bash
cd python && uvx maturin build --release --out ../target/wheels
```

```bash
uv run --with ./target/wheels/<できた wheel> --with pytest python -m pytest python/tests -q
```

`python -m pytest python/tests` はリポジトリのルートから回すこと。`python/` の中から回すと
ソースの `python/ocrus/` が wheel の `ocrus` を隠して `_native が無い` で落ちる。

落ちたときの症状別対応は [references/build-verify.md](references/build-verify.md)。
依存更新が原因の失敗（API 破壊）と、元から壊れていた失敗を区別すること。前者なら
`git diff Cargo.toml scripts/pyproject.toml` に原因のパッケージが写っている。

## 3. OCR が壊れていないか（約 20 秒）

ビルドが通っても認識が空になることがある。前処理・レイアウト・デコードのどこかが
静かに死んでも `cargo test` は緑のままなので、実際に画像を通して確かめる。

```bash
cargo test -p ocrus-engine --release --test smoke -- --nocapture
```

見るのは 3 つ。

- `sample_ja.png -> N line(s)`: 行が取れているか（0 ならレイアウトが死んでいる）
- `non-empty output : N/24`: 空出力ばかりになっていないか
- カテゴリごとの `n/8`: 前回と比べて動いたか

`--release` を付けること。debug ビルドの推論は 100 倍以上遅く、1 枚で数分かかる。

**単文字の正解率はいま 0% だが、これは既知の不具合であって「今回の変更で壊れた」ではない。**
`.ocnn` モデル（または ocrus-nn の実行）が ONNX と違う logits を返すことが実測で確認されている
（同じ入力テンソルで ONNX は 'あ'、`.ocnn` は 'ｏ〗'）。char_accuracy も 0.0% に落ちている。
依存更新の切り分けとしては **0% のままかどうか**と `non-empty` の件数を見ればよい。
不具合の追跡は ocrus-model-improvement の「いま分かっている最大の一手」。

## 4. 精度の回帰はユーザーに依頼する

スモークは「動いている」の確認で、「精度が落ちていない」の確認ではない。
**OCR の結果に効く層を触った回**は、正式な精度測定をユーザーに依頼する。

効く層: `ocrus-preproc` / `ocrus-layout` / `ocrus-recognizer` / `ocrus-nn` / `ocrus-engine`、
および `image` / `imageproc` / `ndarray` / `wide` の更新。
依存が clap や serde だけの回、ドキュメントだけの回は依頼しなくてよい。

依頼するコマンド（そのまま渡す）:

```bash
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
```

```bash
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture
```

step1 が約 36 分、step2 が約 3 分。結果は `logs/char_accuracy_<step>_<epoch>.log` に残るので、
そのパスを返してもらう。返ってきたら変更前のログと突き合わせる。

```bash
python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py step1
```

同じ step・同じ `test_images` なら char_accuracy は決定的に動くので、差はそのままコードの差。
分母（`correct/total` の total）が違う 2 本は比較にならない。スクリプトが警告を出したら、
比較ではなく測り直しが要る。

落ちていたら ocrus-model-improvement へ引き継ぐ。層の切り分けと実験はあちらの仕事。

**測っていない回に「精度は落ちていない」と書かないこと。** 書いてよいのは
「スモークは通った」まで。ここを曖昧にすると、後から誰も検証結果を信用できなくなる。

## 5. コミットする

コミットするのは**このメンテナンスで触ったファイルだけ**。手順 0 の未コミット変更は
そこで別のコミットとして確定させてあるはずで、依存更新のコミットには混ぜない。
理由の違う変更が 1 つのコミットに入ると、revert で切り分けられなくなる。
そのため `git add -A` や `git commit -a` は使わず、必ずパスを列挙する。

```bash
git add Cargo.toml Cargo.lock scripts/pyproject.toml scripts/uv.lock python/pyproject.toml
git commit -m "..."
```

ビルドを通すためにソースへ手を入れた場合（依存の API 破壊への追従など）は、そのファイルも
同じコミットに含める。依存更新とそれへの追従は 1 つの単位として意味を持つ。
一方、精度改善は理由の違う変更なので**コミットを分ける**。

メッセージは既存の履歴に合わせて日本語・Conventional Commits で書く。何を上げたかが
後から追えるよう、主要なパッケージ名とバージョンを本文に残す。

```
chore(deps): 依存パッケージを更新

- image 0.25.9 → 0.25.10
- clap 4.5.60 → 4.6.0

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

更新が無かった回はコミットしない。空コミットは履歴のノイズになる。

**push はしない。** リモートへ出すかはユーザーが決めることで、メンテナンスの完了条件には
含まれない。指示があったときだけ行う。

## 6. 報告する

何をどう直したかより、**次に触る人が知りたいこと**を先に書く。

```
## 依存更新
| 対象 | 更新 | 備考 |
（Rust / scripts の Python / python の順。上げなかったものは理由も）

## ビルドと検査
| build | test | clippy | fmt | maturin | pytest |

## OCR スモーク
sample_ja: N 行 / 単文字: n/24 正解, m/24 非空
（前回値と比べてどうか。動いていないなら「据え置き」と書く）

## 精度
（測っていないなら「未測定」と明記する。依頼したなら渡したコマンドと、
 返ってきたログのパス、compare_accuracy.py の結果）

## コミット
| ハッシュ | 内容 |
（手順 0 で確定させた先行分も含める。承認なしで確定させた以上、何をどういう理由で
 コミットしたかがここだけで分かるようにする）

## 残っている問題
（直せなかったもの、判断を仰ぎたいもの）
```

## 判断に迷ったとき

- **depup がメジャー更新を提案してきた** — 単独で当てて、そのつどビルドとスモークを回す。
  複数を混ぜて当てると、落ちたときにどれが原因か分からなくなる。
- **clippy が既存コードで警告を出す** — 自分が触っていない場所の警告は直さない。
  報告に書いて判断を仰ぐ。ついでの整形は、後から差分を読む人の負担になる。
- **スモークの単文字が 0% のまま** — 現状どおり。手順 3 の注記を読む。
- **スモークの `non-empty` が大きく減った** — 依存更新の疑いが濃い。まず `git diff` で
  `image` / `ndarray` / `wide` が動いていないか見る。
- **精度テストを自分で回したくなった** — 回さない。36 分かかるうえ、セッションが終われば
  結果ごと消える。ユーザーに渡す。
- **テストが落ちたままコミットしてよいか** — しない。ビルド・検査・スモークが通ってから
  コミットする。通せない事情があるなら、コミットせずに報告して指示を仰ぐ。
- **モデルが無い環境だった** — スモークと精度は skip される。`repo_status.py` の出力に
  そう出るので、報告に「モデル未取得のため OCR は未検証」と書く。黙って飛ばさない。
