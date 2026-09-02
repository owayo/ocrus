---
name: ocrus-model-improvement
description: |
  ocrus（Pure Rust の日本語 OCR）の認識精度を上げる作業。char_accuracy のログと
  test_results/failures_*.json を解析して失敗の質（空出力か安定誤認識か）を切り分け、
  todo.md に記録済みの「試して効かなかった手法」を繰り返さずに次の実験を選び、
  同じ条件の A/B で採否を決めて結果を todo.md に残すまでを扱う。
  「精度を上げて」「認識精度が悪い」「char_accuracy の結果を見て」「failures を分析して」
  「この文字が読めない」「モデルを改善して」「TLA を本番に入れて」のような依頼では
  必ずこのスキルを使うこと。
  依存更新・ビルド・コミットまでの定期保守は ocrus-maintenance。
  そちらのスモークが劣化したときも、原因の切り分けはこのスキルで行う。
---

# ocrus の精度改善

このリポジトリで精度改善が失敗する原因は、たいてい 3 つのどれかに落ちる。

1. **過去に効かなかった手法をもう一度試す** — `todo.md` に理由まで残っているのに読んでいない
2. **比較にならない 2 つの数字を比べる** — step が違う、フォント集合が違う、途中で中断している
3. **合計だけ見て採用する** — 狙ったカテゴリは上がっているが別のカテゴリを壊している

順番に潰せるよう、測る前に条件を決めて、測った後は必ず記録する形にしてある。

## 先に読むもの

- `todo.md` — 現在地、試して失敗した手法とその理由、未試行の候補
- [references/failed-experiments.md](references/failed-experiments.md) — 失敗の要約と、
  再挑戦してよい条件
- [references/accuracy-protocol.md](references/accuracy-protocol.md) — 測り方の決まりごと

`todo.md` が一次資料で、references は要約。食い違ったら `todo.md` を信じる。

## 過去にあった落とし穴: 陳腐化した .ocnn

2026-09-02 に step2 が **0.0%**（前回ログは 76.9% / 68.6%）まで落ちていた。
原因は **`~/.ocrus/models/rec.ocnn` が古い変換器で作られた成果物だったこと**。
`rec.onnx` から作り直したら、その場で 76.9% / 68.6% に完全復帰した。

切り分けの手順（同じ症状が出たらこれをなぞる）:

1. Rust が作った入力テンソルをダンプし、**ONNX Runtime に同じものを流す**。
   正しい文字が出るなら前処理は無罪で、モデル成果物か推論エンジンが原因
2. `rec.onnx` から `.ocnn` を作り直し、`OCRUS_MODEL_DIR` を向けて比べる

```bash
uv run --with onnx --with onnxruntime --with numpy \n  python scripts/src/ocrus_scripts/convert_to_ocnn.py \n  ~/.ocrus/models/rec.onnx -o /tmp/model_fresh/rec.ocnn
```

```bash
OCRUS_MODEL_DIR=/tmp/model_fresh cargo test -p ocrus-engine --release --test smoke -- --nocapture
```

**いまは 1 コマンドで分かる。** `.ocnn` には変換時に測ったゴールデン出力が埋まっている。

```bash
cargo test -p ocrus-nn --release --test ocnn_golden -- --nocapture
```

モデルと実行系が食い違っていればここで落ちる。おかしいと思ったら最初にこれを回す。

## 本番を測る（これが最優先）

**本番パイプラインの精度を直接測れる。** 評価経路（`char_accuracy`）ではなく、
利用者が通る `OcrEngine` を 845 枚に通す。約 40 秒なので AI セッションから回してよい。

```bash
cargo test -p ocrus-engine --release --test accuracy -- --ignored --nocapture
```

`OCRUS_ACC_CATEGORIES` と `OCRUS_ACC_FONTS` で絞れる。
2026-09-03 に評価経路との 24.3pt の差を潰し、さらに小書きかなの補正で追い越した（`todo.md` 参照）。
**今後の改善はこれで測る。** char_accuracy は「モデルの素の力」を見る別物。

## 現在地（2026-09-03 実測）

| 指標 | 値 |
| --- | --- |
| 本番パイプライン（かな 845 枚） | ひらがな 92.8% / カタカナ 83.5% / 合計 **88.0%** |
| 常用漢字（2,136 枚・1 フォント） | **97.4%** |
| `--mode accurate` | 88.5%（前処理バリアントの多数決、推論 3 回ぶん） |
| `--charset jis` 併用 | 88.8% |
| step2（評価経路） | 76.9% / 68.6% ← **本番の方が上になった** |
| 本番のスモーク（24 枚） | 12/24、非空 24/24。**枚数が少なすぎるので精度判断には使わない** |
| 推論速度 | W=104 で **62ms**（2026-09-02 のカーネル改善前は 592ms） |
| モデルロード | 1.2ms（`.ocnn` の JSON メタデータ解析込み） |
| モデルサイズ | **40.2MB**（`.ocnn` f16。f32 なら 80.2MB） |
| op ごとの時間 | conv2d 83% / binary 5% / matmul 2%（W=104） |

## 1. 失敗の質を見る

数字ではなく中身を見る。同じ 40% でも、空出力が多い層と安定誤認識が多い層では効く手が違う。

```bash
python .claude/skills/ocrus-model-improvement/scripts/failure_report.py step1
```

```bash
python .claude/skills/ocrus-model-improvement/scripts/failure_report.py step2
```

読み方:

- **空出力が多い（記号系）** — モデルが「何も無い」と判定している。時系列 logit の集約
  （TLA）や前処理バリアントが効いた実績がある層。本番では空出力に TLA を当てて 3 件まで減った。
- **信頼度が高いまま外している** — 失敗リストに信頼度も出る。中央 0.9 を超えるならデコード側の
  工夫では動かない。同形の字（カ↔力、へ↔ヘ）は孤立文字である限り解けないと判断してよい。
- **空出力が少なく混同ペアが集中（かな・CJK）** — 似た文字への安定した誤認識。
  デコード側の小細工では動かない。`ぁ→あ`（字形の大小）や `カ→力`（別字だが同形）は、
  字形の差を残す前処理か、文脈・頻度のような外部情報が要る。

## 2. 実験を決める

手を動かす前に、この 5 つを言葉にする。書けないなら、まだ実験の形になっていない。

1. **仮説** — なぜ効くと考えるのか（失敗の質のどれに効くのか）
2. **触る場所** — どの crate のどの関数か
3. **効くはずのカテゴリ** — step1 の記号だけ、step2 のかなだけ、など
4. **壊れうるカテゴリ** — 副作用が出るとしたらどこか
5. **採否の基準** — 何 pt 上がったら採用、何が下がったら見送りか

`todo.md` の「試して失敗した手法」と同じことをやろうとしていないか、ここで確認する。
再挑戦するなら、**前回の失敗理由のどこが変わったのか**を仮説に含める。
「今度はうまくいくかもしれない」は仮説ではない。

## 3. 測る

精度測定は AI セッションからは実行しない（`AGENTS.md`。step1 は約 36 分かかり、
セッションが終わるとコマンドごと死ぬ）。**変更前の値を控えてから**ユーザーに依頼する。

変更前の控え（失敗リストは実行のたびに上書きされる）:

```bash
cp test_results/failures_step1.json test_results/failures_step1.base.json
```

ユーザーに渡すコマンド:

```bash
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
```

```bash
cargo test -p ocrus-cli --test char_accuracy char_accuracy_step2 --release -- --ignored --nocapture
```

step3 系（常用漢字・JIS 1〜4 水準）はさらに長い。1 つずつ、必要なときだけ依頼する。

## 4. 比べる

```bash
python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py step1
```

```bash
python .claude/skills/ocrus-model-improvement/scripts/failure_report.py step1 \
    --diff test_results/failures_step1.base.json
```

前者はカテゴリごとの増減、後者は「どの文字が直り、どの文字が壊れたか」。
両方見ること。合計が同じでも中身が入れ替わっていることがある。

比較が成立する条件は [references/accuracy-protocol.md](references/accuracy-protocol.md) に
まとめてある。分母が違う 2 本を比べても意味がないので、スクリプトが警告を出したら
測り直す。

### 採否の目安

- **採用** — 狙ったカテゴリが上がり、他のカテゴリが 0.5pt を超えて下がっていない
- **保留** — 上がったが別のカテゴリを壊した。両方の数字を `todo.md` に書いて判断を仰ぐ
- **見送り** — 変化なし、または悪化。**コードは戻すが、記録は残す**

「変化なし」は失敗ではなく情報。同じ手法を半年後にもう一度試さないために、
必ず `todo.md` に残す。

## 5. 記録する

`todo.md` に、思いつきではなく**測った結果**を書く。

```markdown
#### ✅ / ⚪ / ❌ <手法名>
- **原理**: 何をしたか
- **コード**: 触ったファイル
- **結果**: カテゴリごとの数字（before → after, ±pt）
- **結論**: 採用 / 保留 / 見送りと、その理由
- **ログ**: logs/char_accuracy_stepN_*.log
```

効かなかった手法ほど、理由まで書く価値がある。`todo.md` の
「試して失敗した手法と理由」は、このリポジトリで一番価値のある資産になっている。

## 6. 報告する

```
## 何を試したか
（仮説と、触った場所）

## 測定
| カテゴリ | before | after | 差 |
（同じ step・同じフォント集合であることを明記。未測定なら「未測定」と書く）

## 直った文字 / 壊れた文字
（failure_report --diff の要約。代表例を数文字）

## 判断
採用 / 保留 / 見送りと理由

## todo.md の更新
（追記した節）
```

## 判断に迷ったとき

- **どこから手を付けるか** — まず現在の数字を実測する。過去のログは陳腐化した成果物で
  測られている可能性がある（実際にあった）。測り直してから手法を選ぶ。
- **単文字の精度をこれ以上追うべきか** — 単文字は本来の用途ではない。モデルはテキスト行に
  最適化されている。`todo.md` の「D. テキスト行テスト」が未着手のまま残っているので、
  実用精度を知りたいならそちらが先。かなの残り 101 件のうち約半分
  （カ↔力、へ↔ヘ のような同形）は孤立文字である限り解けない。
- **数字が動かない** — 手法が効いていないのか、経路に入っていないのかを先に切り分ける。
  評価コードに入れたつもりで本番に入っていない（またはその逆）は実際に起きている。
- **モデル自体を再学習したくなった** — ファインチューニングは `scripts/` の領分で、
  PaddlePaddle と Python 3.12 が要る。長時間かかるのでユーザーに依頼する。
  現状 `finetune` の export が壊れている（`todo.md` 参照）ので、まず export を直す。
- **前処理を変えたら char_accuracy が全部下がった** — `normalize_line(g, b)` は
  `normalize_line_scaled(g, b, 1.0)` そのものなので、本番と評価は同じ正規化を通る。
  片方だけ直したつもりでも両方動く。
- **時間がかかりすぎる** — step1 全体を毎回回さない。まず対象カテゴリだけを見て、
  採用の判断が近づいてから全体を測る。
