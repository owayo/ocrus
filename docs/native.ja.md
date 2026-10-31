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

既定では実行中のMac向けにビルドします。SwiftPMの `binaryTarget` へ生成物を指定すると、
`import OcrusFFI` で呼べます。モデルは別に配置し、ディレクトリを作成関数へ渡します。

## 呼び出しの契約

- エンジンは `ocrus_engine_create` で作り、`ocrus_engine_free` で解放します。
- RGBA8またはBGRA8の画素を `ocrus_recognize_pixels` へ渡します。行末のパディングは
  `stride` に含めます。画像の一時保存やPNGへの再エンコードは不要です。
- JSONとエラー文字列はRustが確保します。読み終えたら `ocrus_string_free` で解放します。
  出力先はNULLで初期化し、出力先どうしを重ねないでください。
- 同じエンジンへの認識呼び出しは直列にします。Swiftでは専用キューをexecutorにしたactorなどで
  ブロッキング処理を実行します。エンジンごとのrayonプールはグローバルプールを変更しません。
- キャンセルは別スレッドから要求できます。進行中のモデル推論は完了を待ち、次の推論の前に
  中断します。認識とキャンセル要求の両方が終わるまでトークンを保持します。
- Rustのunwindするpanicは境界で捕捉し、ステータス4を返します。そのエンジンは破棄してください。
  メモリ確保失敗など、プロセスをabortする障害は捕捉できません。

ABIバージョンは `ocrus_abi_version()`、現在の値は1です。JSONはCLIと同じ形式です。

## Cヘッダとリンクの検証

macOSでは生成した静的ライブラリに対して次を実行できます。

```bash
cc -I crates/ocrus-ffi/include crates/ocrus-ffi/tests/smoke.c \
  target/release/libocrus_ffi.a -framework CoreFoundation -framework Security \
  -liconv -lresolv -o /tmp/ocrus-ffi-smoke
/tmp/ocrus-ffi-smoke "$HOME/.ocrus/models"
```

Cからの作成・画像入力・JSONの解放・キャンセルを通します。OCR精度の評価には使いません。
