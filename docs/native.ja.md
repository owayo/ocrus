# C・Swiftへの組み込み

`ocrus-ffi` は認識エンジンを同じプロセス内で実行するC ABIです。モデルを読み込んだ
ハンドルで複数の画像を認識できます。認識処理はCLI・Pythonと同じ `ocrus-engine` にあります。

## ビルド

```bash
mise exec -- cargo build --locked --release -p ocrus-ffi
```

`target/release/` に静的ライブラリと共有ライブラリを生成します。
宣言と所有権の契約は [`ocrus.h`](../crates/ocrus-ffi/include/ocrus.h) を参照してください。
macOSでSwiftPMから使う場合は、XCFrameworkを生成します。

```bash
bash scripts/build-xcframework.sh "$PWD/artifacts/OcrusFFI.xcframework"
# IntelとApple Siliconをまとめる場合
OCRUS_ARCHS="arm64 x86_64" bash scripts/build-xcframework.sh "$PWD/artifacts/OcrusFFI.xcframework"
```

既定では実行中のMac向けにビルドし、`accelerate` featureでOS標準のBLASを使います。
追加の推論ランタイムは不要です。SwiftPMのリンク設定に `.linkedFramework("Accelerate")`、
Cのリンクオプションに `-framework Accelerate` を追加してください。
通常のCargoビルドではこのfeatureは無効で、従来のRust演算を使います。

SwiftPMの `binaryTarget` へ生成物を指定すると、
`import OcrusFFI` で呼べます。モデルは別に配置し、ディレクトリを作成関数へ渡します。

## 呼び出しの契約

- エンジンは `ocrus_engine_create` で作り、`ocrus_engine_free` で解放します。
- RGBA8またはBGRA8の画素を `ocrus_recognize_pixels` へ渡します。行末のパディングは
  `stride` に含めます。画像の一時保存やPNGへの再エンコードは不要です。
- JSONとエラー文字列はRustが確保します。読み終えたら `ocrus_string_free` で解放します。
  出力先はNULLで初期化し、出力先どうしを重ねないでください。
- 同じエンジンへの認識呼び出しは直列にします。Swiftでは専用キューをexecutorにしたactorなどで
  ブロッキング処理を実行します。エンジンごとのrayonプールはグローバルプールを変更しません。
- キャンセルは別スレッドから要求できます。最大4行の進行中の推論が完了してから
  中断します。認識とキャンセル要求の両方が終わるまでトークンを保持します。
- Rustのunwindするpanicは境界で捕捉し、ステータス4を返します。そのエンジンは破棄してください。
  メモリ確保失敗など、プロセスをabortする障害は捕捉できません。

ABIバージョンは `ocrus_abi_version()`、現在の値は1です。JSONはCLIと同じ形式です。

## Cヘッダとリンクの検証

macOSでは生成した静的ライブラリに対して次を実行できます。

```bash
cc -I crates/ocrus-ffi/include crates/ocrus-ffi/tests/smoke.c \
  target/release/libocrus_ffi.a -framework CoreFoundation -framework Security -framework Accelerate \
  -liconv -lresolv -o /tmp/ocrus-ffi-smoke
/tmp/ocrus-ffi-smoke "$HOME/.ocrus/models"
```

Cからの作成・画像入力・JSONの解放・キャンセルを通します。OCR精度の評価には使いません。

## 英単語間の空白

認識結果のCTC文字位置と、入力画像の字形間の隙間がともに広い場合、欠けた半角スペースを
1個補います。認識済みの文字列とCTC列が一致する行だけが対象です。空白以外の文字や
既存の空白は変更せず、追加の推論も行いません。

URL・パス・小数などに見える連続部分と全角英数字は補正から除外します。
字下げ、桁揃え用の連続スペース、行末の空白は復元しません。
