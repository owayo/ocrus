# 精度の測り方

## 何を測っているのか

`crates/ocrus-cli/tests/char_accuracy.rs` は、**1 文字だけを描いた画像**を
`test_images/<フォント>/<カテゴリ>/U+XXXX.png` から読み、認識結果の最初の非空白文字が
期待した文字と一致するかを数える。

- 比較は NFKC 正規化を通した上で行う（`chars_match`）
- モデルは `$OCRUS_MODEL_DIR`、既定は `~/.ocrus/models` の `rec.ocnn`
- 正規化は `normalize_line_scaled(g, b, 1.0)`（= 本番の `normalize_line` と同一）、
  デコードは greedy と TLA の併用。本番との差はデコードだけ

**モデル成果物が変われば数字も変わる。** 2026-09-02 に step2 が 0.0% まで落ちていたが、
原因は陳腐化した `rec.ocnn` で、`rec.onnx` から作り直したら 76.9% / 68.6% に完全復帰した。
過去のログと比べる前に、`rec.ocnn` が測定当時と同じものかを確かめること。

## step とカテゴリ

| step | カテゴリ | 文字数の目安 | 所要 |
| --- | --- | --- | --- |
| step1 | halfwidth_alnum, halfwidth_symbols, fullwidth_alnum, fullwidth_symbols | 219 字 × 5 フォント | 約 36 分 |
| step2 | hiragana, katakana | 169 字 × 5 フォント | 約 3 分 |
| step3_joyo | joyo_kanji | 2,136 字 | 長い |
| step3_jis1〜4 | jis_level1〜4 | さらに長い | 非常に長い |

step3 系は 1 つずつ、必要なときだけ回す。

## 出力

| ファイル | 中身 | 上書き |
| --- | --- | --- |
| `logs/char_accuracy_<step>_<epoch>.log` | 進捗とカテゴリ別集計、末尾に Overall Accuracy | されない（毎回新規） |
| `test_results/failures_<step>.json` | 失敗した文字の一覧（文字・カテゴリ・フォント・認識結果） | **される** |

失敗リストは実行のたびに上書きされるので、**変更前に控えを取る**。

```bash
cp test_results/failures_step1.json test_results/failures_step1.base.json
```

Ctrl+C で中断した場合、失敗リストはそこまでの分が保存されるが、ログに
`=== Overall Accuracy ===` が残らない。中断したログは比較に使えない。

## 比較が成立する条件

char_accuracy は**決定的**に動く（同じ画像・同じモデル・同じコードなら同じ結果）。
だから 2 本の差はそのままコードの差になる。ただし次が揃っているときだけ。

- 同じ step（step1 と step2 は別物）
- 同じ `test_images/`（フォントを足したら分母が変わる）
- 同じ `data/test_chars/`（カテゴリのファイルを触ったら分母が変わる）
- 同じモデル（`rec.ocnn` を差し替えたら別の測定）
- どちらも最後まで走っている

`compare_accuracy.py` は分母が食い違うと警告を出す。出たら比較せず測り直す。

```bash
python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py step1
python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py --list
```

## 数字の読み方

- **0.1pt 未満の差は「変わっていない」**。決定的に動く以上、本当に同じ結果なら差は 0.0pt に
  なる。0.3pt 程度の差は 1〜3 文字の増減で、たまたまの可能性がある
- **合計だけで判断しない**。狙ったカテゴリ以外が下がっていないかを必ず見る
- **同じ pt でも意味が違う**。半角記号は 160 字しかないので 1 文字が 0.6pt 動く。
  母数の小さいカテゴリの派手な変動は過大評価しない
- **`correct/total` を先に見る**。パーセントは丸められている

## A/B（FP32 と量子化モデル）

`OCRUS_QUANTIZED_MODEL` を設定すると、同じテストで 2 つのモデルを同時に測って
`FP32=... INT8=... diff=...` の形でログに出る。片方ずつ 2 回回すより速く、
条件も完全に揃う。量子化の影響を見るときはこちらを使う。

```bash
OCRUS_QUANTIZED_MODEL=path/to/rec_int8.ocnn \
  cargo test -p ocrus-cli --test char_accuracy char_accuracy_step1 --release -- --ignored --nocapture
```

## 短いスモークとの違い

`cargo test -p ocrus-engine --release --test smoke -- --nocapture` は
**本番パイプライン**を 24 枚だけ通す。約 20 秒で終わる。基準値は 12/24 正解・23/24 非空
（2026-09-02）。精度の指標ではなく、壊れていないことの確認に使う。
