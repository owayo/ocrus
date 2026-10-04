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

学習・モデル変換・大規模な精度テストは別途実行します。コマンドは [学習](training.ja.md)、過去の検証結果は [保守記録](maintenance-2026-10-04.md) を参照してください。

画像生成・精度テストのデータ置き場は `OCRUS_DATA_DIR` で指定できます。未設定ならリポジトリのルートを使います。相対パスはリポジトリのルートを基準に解決します。`test_images/`、`test_results/`、画像生成に使う `fonts/` が対象です。文字一覧と `test_fonts.yml` はリポジトリ内のものを使い、実行ログは `logs/` に書き出します。
