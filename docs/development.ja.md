# 開発

ツールの版は `mise.toml` に固定し、手元と CI の検査は Makefile にまとめています。

```bash
make setup      # ツールと依存をインストール
make ci         # 書式・clippy・コンパイル・テストを確認
make build      # 全クレートをビルド
make release    # CLI のリリースバイナリをビルド
make fmt        # Rust コードを整形
make bench      # ベンチマーク
```

`make test` は macOS の Python ライブラリ探索パスも設定します。モデルがない環境では OCR の実行テストが自動で省略されるため、単体テストがすべて成功しても本番のモデルで文字を正しく認識できることまでは確認できません。

Python wheel をビルドし、その wheel と pytest をテスト用の Python 環境にインストールしてから、リポジトリのルートでテストを実行します。

```bash
make wheel
make pytest
```

`python/` 内で pytest を実行すると、インストール済み wheel よりもソースパッケージが優先されます。workspace のテストで Python とリンクできるよう、`pyo3/extension-module` はクレートの既定 feature に含めません。

OCR のスモークテストにはモデルが必要です。リリースビルド後の実行時間は約 20 秒です。

```bash
make smoke
```

学習・モデル変換・大規模な精度テストは別途実行します。コマンドは [学習](training.ja.md)、検証結果は [2026-10-29 の保守記録](maintenance-2026-10-29.md)、[2026-10-26 の保守記録](maintenance-2026-10-26.md)、[2026-10-18 の保守記録](maintenance-2026-10-18.md)、[2026-10-04 の保守記録](maintenance-2026-10-04.md) を参照してください。

画像生成・精度テストのデータ置き場は `OCRUS_DATA_DIR` で指定できます。未設定ならリポジトリのルートを使います。相対パスはリポジトリのルートを基準に解決します。`test_images/`、`test_results/`、画像生成に使う `fonts/` が対象です。文字一覧と `test_fonts.yml` はリポジトリ内のものを使い、実行ログは `logs/` に書き出します。

## 辞書補正の速度測定

合成文字列を使うため、モデルや評価画像は不要です。短い認識行と長文のそれぞれで、
ルールが一致する場合と一致しない場合を測ります。

```bash
mise exec -- cargo bench --locked --bench dict -- --save-baseline before
# 比較する変更を加えた後、同じツールチェーンで実行する
mise exec -- cargo bench --locked --bench dict -- --baseline before
```

測定結果は `target/criterion/dict_correct/` に保存されます。

モデルのロード・推論内訳と、行検出・辞書補正のメモリ確保は、次の example で CSV に出力できます。
計数用アロケータの負荷を含むため、同じ計測方法で比較してください。

```bash
mise exec -- cargo run --release --locked --example maintenance_profile -- model.ocnn
```

## 水平射影の行検出の速度測定

合成二値画像を使い、24×33、600×800、600×803、2000×3000（高さ×幅）の4条件を測ります。
小画像、16画素の SIMD 幅に満たない端数、大きいページでの退行を比較できます。

```bash
mise exec -- cargo bench --locked --bench preproc -- layout_detect_lines --save-baseline before
# 比較する変更を加えた後、同じ入力・ビルド条件で実行する
mise exec -- cargo bench --locked --bench preproc -- layout_detect_lines --baseline before
```

`detect_lines_projection` は、行が連続した配列には SIMD、列優先や逆順ストライドの配列には
画素イテレータを使い、行を採用する前景数に達したら残りの走査を省きます。
メモリ配置が異なっても、同じ画素に対して同じ行境界を返します。
