# ocrus OCR精度改善 TODO

## 現在の状態

### 完了済み
- [x] PP-OCRv5公式モデル → ONNX変換（511ノード、`/tmp/ppocrv5_official.onnx`）
- [x] ONNX → OCNN変換（369レイヤー、`/tmp/ppocrv5_official.ocnn`）
- [x] モデルインストール（`~/.ocrus/models/rec.ocnn` + `rec.onnx` + `dict.txt`）
- [x] PAD_VALUE修正: 1.0 → -1.0（PaddleOCRのゼロパディングに合致）
- [x] MIN_WIDTH=320 廃止 → WIDTH_ALIGN=8 に変更（パディング93%→最大15%に削減）
- [x] Nearest-neighbor → Bilinear補間に変更
- [x] `normalize_line_scaled()` 関数追加（幅スケーリング対応）
- [x] WSL環境でのbuild-essential, Rustインストール

### 精度結果（公式PP-OCRv5、1文字単体認識）

| カテゴリ | 初期(MIN_WIDTH=320,NN) | 改善後(ALIGN=8,bilinear) | 改善幅 |
|----------|----------------------|------------------------|--------|
| 半角英数 | 69.7% | **76.1%** | +6.4pt |
| 半角記号 | 28.7% | **31.9%** | +3.2pt |
| 全角英数 | 61.6% | **77.1%** | +15.5pt |
| 全角記号 | 25.1% | **35.2%** | +10.1pt |
| ひらがな | 71.3% | **76.9%** | +5.6pt |
| カタカナ | 59.1% | **68.6%** | +9.5pt |

### 2026-03-10 追加評価結果（TLA / Top-K beam / 画像特徴補正）

#### TLA (Temporal Logit Aggregation) 実測
- `cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture`
  - 半角英数: **76.5%**（76.1% → +0.4pt）
  - 半角記号: **38.1%**（31.9% → +6.2pt）
  - 全角英数: **77.1%**（据え置き）
  - 全角記号: **39.7%**（35.2% → +4.5pt）
- `cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture`
  - ひらがな: **76.9%**（据え置き）
  - カタカナ: **68.6%**（据え置き）
- **結論**: TLAは「空出力寄りの記号カテゴリ」に明確に効く。かな系にはほぼ効かない。
- **ログ**:
  - `logs/char_accuracy_step1_1773072063.log`
  - `logs/char_accuracy_step2_1773072069.log`

#### Top-K Pruned Beam Search 実測
- `ocrus-recognizer` に `ctc_beam_decode_topk()` を実装し、`char_accuracy` の曖昧ケース救済に組み込み評価
- **結果**: `step1` 総合値は TLA 単独時と完全に同一
  - 半角英数 76.5%, 半角記号 38.1%, 全角英数 77.1%, 全角記号 39.7%
- **結論**: 単文字ベンチでは追加効果なし。APIは残すが、現時点ではデフォルト採用しない。
- **ログ**: `logs/char_accuracy_step1_1773072985.log`

#### 混同ペア Post-Correction + 画像特徴量 実測
- 小書きかな補正を `char_accuracy` に試験実装し、foreground bbox 比率で `あ→ぁ` 系を補正
- **結果**: `step2` 総合値は変化なし
  - ひらがな 76.9%, カタカナ 68.6%
- **結論**: しきい値ベースの単純な画像特徴補正では改善しなかった。いったん撤回。
- **ログ**: `logs/char_accuracy_step2_1773072757.log`

### 失敗パターン分析
- **空出力 (29%)**: 記号で多い。モデルが認識不能（], [, ,, ^, §, ※など）
- **小書き文字混同**: ぁ→あ, ッ→ツ, ょ→よ（resize後にサイズ差が消失）
- **視覚類似CJK**: カ→力, ハ→八, エ→工（コンテキストなしでは区別不可）
- **大文字小文字/半角全角**: s→Ｓ, z→Ｚ（モデルが全角大文字を優先）

### 2026-03-10 時点の failure 分析
- `step1` failures: 433件、うち空出力 20件（4.6%）
  - 主要混同: `s→Ｓ`, `z→Ｚ`, `c→Ｃ`, `<→く`, `.→a`, `-→a`, `［→「`, `］→」`, `※→""`, `§→""`
- `step2` failures: 231件、うち空出力 3件（1.3%）
  - 小書きかな系の失敗が **100件**
  - 主要混同: `ぁ→あ`, `ぃ→い`, `ゎ→わ`, `ゃ→や`, `ゅ→ゆ`, `ょ→よ`, `っ→つ`, `ァ→ア`, `カ→力`, `ェ→工`
- **解釈**:
  - 記号は blank 支配やノイズに近い誤りが多く、TLA が効いた
  - かなは空出力ではなく「似た文字への安定誤認識」なので、TLA では改善しない

## 🔴 現在の最重要課題: 新しい精度向上手法の発見

### 背景
前処理の改善（bilinear補間、パディング削減）で+5〜15pt改善したが、まだ半角英数76%・カタカナ69%と不十分。
**従来の手法（Multi-width Ensemble、CTC Beam Search）では効果が出なかった。**
まったく新しいアプローチが必要。

### 試して失敗した手法と理由

#### ❌ Multi-width Ensemble（複数スケールで推論→信頼度投票）
- スケール [1.0, 1.5, 2.0, 3.0] で推論し、最高信頼度の結果を採用
- **結果**: 精度変化なし or 低下
- **失敗理由**: softmax信頼度がスケール間で比較不適。自然幅(1.0)のパディングなし推論が常に最高信頼度を返す。非blankタイムステップのみの信頼度も試したが、今度は大きいスケールを不当に優先して精度が低下(76.1%→56.5%)
- **教訓**: CTC logitベースの信頼度は異なる入力サイズ間での比較に使えない

#### ❌ CTC Beam Search（beam幅10）
- `ctc_beam_decode` を multi-width ensemble と組み合わせ
- **結果**: 精度大幅低下（76.1%→24.2%）、速度16倍悪化（0.18s→2.96s/char）
- **失敗理由**: 18385クラスでbeam searchすると計算量が爆発。しかもbeam searchの確率計算がgreedyと異なるため、間違った候補を選択
- **教訓**: 大規模クラス数のCTCモデルでbeam searchするにはTop-K pruningが必須

#### ⚠️ Retry-on-Empty（空出力時のみ大きいスケールでリトライ）
- scale=1.0が空出力の場合のみ scale=2.0, 3.0 を試行
- **結果**: 半角記号 +1.2pt（31.9%→33.1%）のみ微改善
- **教訓**: 空出力の大半はモデルが根本的に認識できない文字。スケール変更では解決しない

### 🟢 実装済み・評価済みの手法

#### ✅ TLA (Temporal Logit Aggregation)
- **原理**: CTC greedyは各timestepでargmaxを取るが、TLAは全timestepのsoftmax確率を文字クラスごとに集約し、最大票の文字を返す
- **コード**:
  - `crates/ocrus-recognizer/src/ctc_tla.rs`
  - `crates/ocrus-cli/tests/char_accuracy.rs`
- **結果**: 記号カテゴリに有効。特に半角記号 +6.2pt、全角記号 +4.5pt
- **現状**: `char_accuracy` の評価経路では採用済み。次は CLI/本番パイプラインへ昇格する価値がある

#### ⚪ Top-K Pruned Beam Search
- **コード**:
  - `crates/ocrus-recognizer/src/ctc_beam.rs`
  - `ctc_beam_decode_topk()` を追加
- **結果**: 単文字ベンチでは追加改善なし
- **現状**: APIは残す。テキスト行評価か別の reranking と組み合わせるまで保留

#### ⚪ 混同ペア Post-Correction + 画像特徴量
- **内容**: 小書きかな向けに foreground bbox 比率ベース補正を試験導入
- **結果**: `step2` で改善なし
- **現状**: いったん不採用。より強い特徴量設計が必要

### 🔵 未試行の有望なアプローチ

#### A. 前処理バリアント Ensemble（画像レベル）
- 同じ入力に対し複数の前処理バリアント:
  1. 通常（現在のbilinear + PAD=-1.0）
  2. 膨張（dilation: 細い文字を太くする → `,` `.` のような細線文字に有効）
  3. 収縮（erosion: 太い文字を細くする → ストローク認識精度向上）
  4. コントラスト強調
- 各バリアントで推論し、CTC/TLA 結果の一致数で採用
- **理由**: 記号系は空出力と細線潰れが主因なので、今の failure 分析に最も整合的

#### B. 文字周辺コンテキスト注入
- 1文字単体ではなく、テスト的に「あ{対象文字}い」のような3文字画像を生成
- モデルがテキスト行認識に最適化されているため、周辺に既知の文字があると精度向上する可能性
- テスト画像生成側（`generate_test_images.rs`）の変更が必要
- **理由**: `カ↔力`, `エ↔工` のようなコンテキスト依存誤りに対し、最も本質的

#### C. Logit Top-K Reranking with Character Frequency Prior
- CTC greedyのargmaxではなく、各timestepのtop-5候補を抽出
- 日本語文字頻度テーブル（例: カ>力、ハ>八 in general Japanese text）で重み付け
- 「形が似ているが出現頻度が大きく異なる文字ペア」で効果的
- **理由**: かな/CJK 混同の多くが frequency prior で押し切れる可能性がある

#### D. テキスト行テスト（本来の用途での評価）
- モデルの本来の用途はテキスト行認識。1文字テストは厳しい評価条件
- 複数文字テスト行を生成して精度評価すると、より実用的な数値が得られる
- 「東京都千代田区」「Hello World 123」のようなテスト行で評価
- `crates/ocrus-cli/tests/` に新テストファイル追加
- **理由**: Top-K beam や frequency prior は単文字より行文脈で効く可能性が高い

### 🔵 まだ深掘りしていない補助案

#### E. Top-K Pruned Beam Search の再利用
- 各timestepで上位K個（例: K=50）のクラスのみ残してbeam search
- 18385→50クラスで速度改善、かつbeam searchの探索精度は維持
- `crates/ocrus-recognizer/src/ctc_beam.rs` を修正
- **メモ**: 単文字では改善なし。行文脈や frequency prior と併用するなら再評価余地あり

### 推奨する優先順位
1. **D. テキスト行テスト** — 実用精度の把握が最優先。単文字では見えない改善余地を確認する
2. **A. 前処理バリアント Ensemble** — TLA が効いた記号カテゴリに対して最も筋が良い
3. **C. Logit Top-K + 文字頻度** — `カ↔力`, `エ↔工` のような安定誤認識を崩せる可能性がある
4. **B. 文字周辺コンテキスト注入** — 単文字ベンチ専用だが、誤認識原因の切り分けに有効
5. **E. Top-K Pruned Beam Search の再利用** — 単体では効かなかったので優先度を下げる

## その他の未完了タスク

### Step3（漢字）テスト実行
- 常用漢字、JIS第1-4水準のテストは未実行
- `cargo test -p ocrus-cli --test char_accuracy char_accuracy_step3_joyo --release -- --ignored --nocapture`

### Finetuned モデルの修正
- 自前finetuneモデルのPaddle export が壊れている（推論出力が間違い）
- 公式PP-OCRv5は正常動作
- 要調査: finetuneスクリプトのexport部分（`scripts/src/ocrus_scripts/`）

### clippy警告の修正
- `char_accuracy.rs` の未使用import警告は解消済み
- 他 crate の clippy は未確認

## ファイル構成

### 変更済みファイル
- `crates/ocrus-preproc/src/normalize.rs` - bilinear補間、WIDTH_ALIGN=8、PAD_VALUE=-1.0、normalize_line_scaled追加
- `crates/ocrus-preproc/src/lib.rs` - normalize_line_scaled のpub export追加
- `crates/ocrus-cli/tests/char_accuracy.rs` - TLA を使う評価経路に更新
- `crates/ocrus-recognizer/src/ctc_tla.rs` - TLA デコーダ追加
- `crates/ocrus-recognizer/src/ctc_beam.rs` - `ctc_beam_decode_topk()` 追加
- `crates/ocrus-recognizer/src/lib.rs` - TLA / Top-K beam export追加

### モデルファイル
- `~/.ocrus/models/rec.ocnn` - PP-OCRv5公式モデル（OCNN形式、81MB）
- `~/.ocrus/models/rec.onnx` - PP-OCRv5公式モデル（ONNX形式、81MB）
- `~/.ocrus/models/dict.txt` - 文字辞書（18383行）
- `/tmp/ppocrv5_official.onnx` - 変換元ONNX
- `/tmp/ppocrv5_official.ocnn` - 変換元OCNN

### 重要な定数
- `TARGET_HEIGHT = 48` - モデル入力高さ（PP-OCRv5固定）
- `WIDTH_ALIGN = 8` - 幅アライメント（モデルstride）
- `PAD_VALUE = -1.0` - パディング値（PaddleOCR準拠: 黒=0→正規化後-1.0）
- `NUM_CHANNELS = 3` - RGB 3チャネル（グレースケールを3ch複製）
- 正規化: `(pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0`
- CTC: blank=index 0, dict.txt 1行目=index 1
- モデル出力: `[1, T, 18385]` where T = width/8

### ビルド・テストコマンド
```bash
# preproc単体テスト
cargo test -p ocrus-preproc --release

# 精度テスト（step1: 英数記号、step2: ひらがなカタカナ）
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture

# ONNX → OCNN 変換
uv run python src/ocrus_scripts/convert_to_ocnn.py input.onnx -o output.ocnn

# Paddle → ONNX 変換
scripts/.venv/bin/paddle2onnx --model_dir <dir> --model_filename inference.json --params_filename inference.pdiparams --save_file output.onnx --opset_version 14 --enable_onnx_checker True
```

### WSL注意事項
- cargoパス: `/home/owayo/.cargo/bin/cargo`
- claw-hooks設定: `~/.claude/claw-hooks/config.toml`（cargoは絶対パス指定）
- git: WSL側のgitを使用（Windows側git.exeだと全ファイルstageされる問題あり）
