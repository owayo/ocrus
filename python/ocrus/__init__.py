"""ocrus - Pure Rust の日本語 OCR エンジンを Python から使う。

CLI (`ocrus recognize`) と同じパイプラインをそのまま呼ぶ。認識結果の JSON も
`OcrResult.to_json()` と CLI の `--format json` で同じ形になる。

    import ocrus

    engine = ocrus.OcrEngine()          # モデルを 1 度だけ読む
    result = engine.recognize("page.png")
    print(result.full_text())

モデルの読み込みは 80MB の mmap と 18,383 行の辞書ロードを伴うので、
**エンジンは使い回すこと**。1 画像ごとに作り直すと、認識そのものより読み込みが重い。

モデルは既定で `~/.ocrus/models`（`OCRUS_MODEL_DIR` で上書き可）から探す。
`rec.ocnn` と `dict.txt` が要る。無ければ [ModelNotFoundError] が上がる。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from . import _native
from ._native import (
    BBox,
    ConfigError,
    ImageError,
    ModelError,
    ModelNotFoundError,
    OcrResult,
    OcrusError,
    Page,
    RubyAnnotation,
    TextLine,
    default_model_dir,
    models_ready,
)

__all__ = [
    "BBox",
    "ConfigError",
    "ImageError",
    "ModelError",
    "ModelNotFoundError",
    "OcrEngine",
    "OcrResult",
    "OcrusError",
    "Page",
    "RubyAnnotation",
    "TextLine",
    "__version__",
    "default_model_dir",
    "models_ready",
    "recognize",
]

__version__: str = _native.__version__

Mode = Literal["auto", "fastest", "accurate"]
Charset = Literal["full", "jis"]

#: 画像として受け取れるもの。パス、エンコード済みバイト列、生ピクセル
#: (numpy 配列 / PIL Image / buffer protocol を持つ任意のオブジェクト)。
ImageInput = Any


class OcrEngine:
    """読み込み済みの OCR エンジン。

    モデル・文字辞書・補正辞書を構築時に読み、以降の認識で使い回す。
    スレッド安全で、認識中は GIL を解放するので、複数スレッドから同じ
    エンジンを呼べば実際に並列に走る。
    """

    __slots__ = ("_engine",)

    def __init__(
        self,
        *,
        model_dir: str | os.PathLike[str] | None = None,
        mode: Mode = "auto",
        charset: Charset = "full",
        dict_path: str | os.PathLike[str] | None = None,
        threads: int | None = None,
        ruby: bool = False,
        beam_width: int = 5,
        confidence_threshold: float = 0.5,
    ) -> None:
        """エンジンを構築する。

        Args:
            model_dir: モデルディレクトリ。既定は `OCRUS_MODEL_DIR`、
                無ければ `~/.ocrus/models`。
            mode: `auto` は画質から経路を決める。`fastest` は品質ゲートを飛ばして
                バッチ推論、`accurate` は CCL レイアウト解析を使う。
            charset: `jis` にすると JIS 文字集合の外のクラスを logit マスクで落とす。
                日本語だけを扱うなら誤検出が減る。
            dict_path: 後処理で使う補正辞書 (`誤り<TAB>正しい` 形式)。
            threads: 行の正規化に使うスレッド数。
            ruby: ルビ（ふりがな）を本文から分離する。
            beam_width: 低信頼行に対する beam search の幅。1 なら使わない。
            confidence_threshold: この値を下回った行で beam search を試す。

        モデルが揃っていなければ `ModelNotFoundError`、`mode` / `charset` の値が
        不正なら `ConfigError` が上がる。

        """
        self._engine = _native.OcrEngine(
            model_dir=Path(model_dir) if model_dir is not None else None,
            mode=mode,
            charset=charset,
            dict_path=Path(dict_path) if dict_path is not None else None,
            threads=threads,
            ruby=ruby,
            beam_width=beam_width,
            confidence_threshold=confidence_threshold,
        )

    @property
    def model_dir(self) -> Path:
        """読み込んだモデルディレクトリ。

        Returns:
            モデルを探したディレクトリ。

        """
        return Path(self._engine.model_dir)

    def recognize(self, image: ImageInput) -> OcrResult:
        """画像を認識する。渡せるものは 4 種類。

        - `str` / `os.PathLike`: 画像ファイルのパス（mmap で読む）
        - `bytes` / `bytearray` / `memoryview`: PNG や JPEG のバイト列
        - numpy 配列: `(H, W)` / `(H, W, 3)` / `(H, W, 4)` の uint8
        - PIL の `Image`: `L` / `RGB` / `RGBA` に変換して渡す

        Args:
            image: 上のいずれか。

        Returns:
            認識結果。

        画像を読めない・デコードできない場合は `ImageError`、対応していない型を
        渡した場合は `TypeError` が上がる。

        """
        if isinstance(image, (str, os.PathLike)):
            return self.recognize_path(image)
        if isinstance(image, (bytes, bytearray, memoryview)):
            return self.recognize_bytes(image)
        return self.recognize_array(image)

    def recognize_path(self, path: str | os.PathLike[str]) -> OcrResult:
        """画像ファイルを認識する。

        Args:
            path: 画像ファイルのパス。

        Returns:
            認識結果。

        """
        return self._engine.recognize_path(Path(path))

    def recognize_bytes(self, data: bytes | bytearray | memoryview) -> OcrResult:
        """エンコード済み画像（PNG / JPEG など）を認識する。

        Args:
            data: 画像ファイルの中身そのもの。

        Returns:
            認識結果。

        """
        return self._engine.recognize_bytes(bytes(data))

    def recognize_raw(
        self,
        data: bytes | bytearray | memoryview,
        width: int,
        height: int,
        channels: int = 1,
    ) -> OcrResult:
        """生ピクセルを認識する。

        Args:
            data: `width * height * channels` バイト、行優先。
            width: 画像の幅。
            height: 画像の高さ。
            channels: 1 (グレースケール) / 3 (RGB) / 4 (RGBA)。

        Returns:
            認識結果。

        """
        return self._engine.recognize_raw(bytes(data), width, height, channels)

    def recognize_array(self, array: Any) -> OcrResult:
        """配列や PIL の Image を認識する。

        numpy は必須依存ではない。配列は `shape` と `tobytes()` だけを見るので、
        同じ形を持つオブジェクトなら何でも受け取れる。

        Args:
            array: `(H, W)` / `(H, W, 3)` / `(H, W, 4)` の uint8 配列、
                または PIL Image。

        Returns:
            認識結果。

        配列とみなせないものは `TypeError`、形や dtype が合わないものは
        `ImageError` になる。

        """
        data, width, height, channels = _as_raw_pixels(array)
        return self.recognize_raw(data, width, height, channels)

    def __repr__(self) -> str:
        """デバッグ用の表示。

        Returns:
            モデルディレクトリを含む文字列。

        """
        return f"OcrEngine(model_dir={str(self.model_dir)!r})"


def recognize(image: ImageInput, **kwargs: Any) -> OcrResult:
    """1 枚だけ認識する便利関数。

    毎回エンジンを作り直すので、複数枚を処理するなら [OcrEngine] を作って
    使い回すこと（モデル読み込みが認識より重い）。

    Args:
        image: [OcrEngine.recognize] が受け取れるもの。
        **kwargs: [OcrEngine] のコンストラクタ引数。

    Returns:
        認識結果。

    """
    return OcrEngine(**kwargs).recognize(image)


def _as_raw_pixels(array: Any) -> tuple[bytes, int, int, int]:
    """配列 / PIL Image / buffer から生ピクセルを取り出す。

    Args:
        array: 変換元。

    Returns:
        (バイト列, 幅, 高さ, チャンネル数)。

    Raises:
        TypeError: 生ピクセルとして解釈できない。

    """
    # PIL Image: mode でチャンネル数が決まる。対応外の mode は L に落とす。
    mode = getattr(array, "mode", None)
    if mode is not None and hasattr(array, "tobytes") and hasattr(array, "size"):
        if mode not in ("L", "RGB", "RGBA"):
            array = array.convert("L")
            mode = "L"
        width, height = array.size
        channels = {"L": 1, "RGB": 3, "RGBA": 4}[mode]
        return array.tobytes(), int(width), int(height), channels

    shape: Sequence[int] | None = getattr(array, "shape", None)
    if shape is None or not hasattr(array, "tobytes"):
        raise TypeError(
            "expected a path, bytes, a numpy array or a PIL Image, "
            f"got {type(array).__name__}"
        )

    # uint8 の連続配列でなければ numpy に直してもらう。numpy が無い環境では
    # そのまま渡し、バイト数が合わなければ ImageError になる。
    dtype = getattr(array, "dtype", None)
    if dtype is not None and str(dtype) != "uint8":
        array = _to_uint8(array)
        shape = array.shape

    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3:
        height, width, channels = shape
    else:
        raise TypeError(f"expected a 2D or 3D array, got shape {tuple(shape)}")

    return array.tobytes(), int(width), int(height), int(channels)


def _to_uint8(array: Any) -> Any:
    """配列を連続した uint8 に直す。

    Args:
        array: 変換元。

    Returns:
        uint8 の連続配列。

    Raises:
        TypeError: numpy が無く、変換できない。

    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - numpy 無し環境向け
        raise TypeError(
            f"array dtype {getattr(array, 'dtype', '?')} is not uint8 and numpy is not "
            "installed to convert it; pass a uint8 array"
        ) from exc
    return np.ascontiguousarray(array, dtype=np.uint8)
