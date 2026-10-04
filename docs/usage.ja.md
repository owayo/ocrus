# 使い方

## 文字認識

```bash
# 基本的な認識
ocrus recognize image.png

# JIS X 0208 文字セット（日本語の誤認識を低減）
ocrus recognize image.png --charset jis

# 辞書による後処理補正
ocrus recognize image.png --dict corrections.txt

# 最速モード（品質ゲートをスキップ、バッチ推論）
ocrus recognize image.png --mode fastest

# 高精度モード（フル品質パイプライン）
ocrus recognize image.png --mode accurate

# ルビ（ふりがな）分離
ocrus recognize image.png --ruby

# カスケード認識（カスケード分類器モデルが必要）
ocrus recognize image.png --cascade path/to/cascade_model.ocnn
```

## Python から使う

```python
import ocrus

engine = ocrus.OcrEngine()              # モデルは 1 度だけ読む（使い回す）
result = engine.recognize("page.png")   # パス / バイト列 / numpy 配列 / PIL Image

print(result.full_text())
for line in result.pages[0].lines:
    print(line.bbox.as_tuple(), round(line.confidence, 3), line.text)
```

CLI と同じパイプラインを呼び、`to_json()` は `--format json` と同じ形で結果を返します。numpy のインストールは任意です。配列を渡すときは `shape` と `tobytes()` を使って読み取るため、numpy への直接依存はありません。詳しい API とインストール方法は [Python パッケージ](../python/README.md) を参照してください。

```bash
make wheel   # wheel をビルド
```

## 対話型 TUI

```bash
ocrus tui
```

ターミナル UI メニューから以下の操作を実行できます：

- E2E 精度テスト
- モデルダウンロード
- ONNX → .ocnn 変換
- データセット生成
- ファインチューン
- ONNX エクスポート
- INT8 量子化
- ベンチマーク

操作方法：`j`/`k` または上下キーで移動、`Enter` で実行、`q` で終了

## ベンチマーク

```bash
ocrus bench image.png -n 100
```

反復回数 `-n` は 1 以上を指定します。データ生成でも `--chars-per-image` は 1 以上とし、検証比率を設定する `--val-ratio` には 0〜1 の有限値を渡してください。0 と 1 も指定できます。不正な値はエラーとして扱います。

## 訓練データ生成

```bash
# システムフォントから訓練データを生成
ocrus dataset generate --output ./training_data --categories hiragana,katakana

# フォントスタイルでフィルタ
ocrus dataset generate --output ./training_data \
  --categories hiragana,katakana --font-styles mincho,gothic

# 追加フォントと小さい文字サイズで生成（指定フォルダだけを使用）
ocrus dataset generate --output ./training_data/small-text \
  --font-dirs ./fonts/screen --no-system-fonts \
  --font-styles gothic,monospace --render-heights 12,16,20,24,32,48

# テスト失敗結果から生成（事前生成済みテスト画像を使用、推奨）
ocrus dataset from-failures --failures ./test_results/failures_step1.json \
  --test-images ./test_images --output ./training_data

# テスト失敗結果からフォント再レンダリングで生成
ocrus dataset from-failures --failures ./failures.json --output ./training_data --all-fonts
```

フォントフォルダは再帰検索し、TTC / OTC 内の各書体を扱います。未収録文字は描画前に除外します。`--render-heights` は描画画像の高さを指定するオプションで、既定値は 48 です。小さい高さを指定した画像もその解像度で保存するため、データ生成時には拡大しません。一般的なフォント選定基準と精度評価は [小さい文字の学習と評価](training-small-text.ja.md) を参照してください。
