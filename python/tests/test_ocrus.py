"""ocrus の Python バインディングのテスト。

モデル (`rec.ocnn` / `dict.txt`) が無い環境では、実際に認識するテストは skip する。
バインディングの形（型・例外・引数の受け取り）だけはモデル無しでも確かめられるので、
そちらは常に走らせる。

リポジトリのルートから回すこと。python/ の中から回すと、ソースの python/ocrus が
インストール済みの ocrus を隠して `_native が無い` で落ちる。

    python -m pytest python/tests -q
"""

from __future__ import annotations

import json
from pathlib import Path

import ocrus
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "testdata" / "sample_ja.png"

needs_model = pytest.mark.skipif(
    not ocrus.models_ready(),
    reason=f"model not found in {ocrus.default_model_dir()}",
)
needs_sample = pytest.mark.skipif(
    not SAMPLE.is_file(), reason="testdata/sample_ja.png がない"
)


def test_version_is_exposed() -> None:
    """バージョンが Cargo.toml から引き継がれている。"""
    assert ocrus.__version__.count(".") >= 1


def test_error_hierarchy() -> None:
    """例外は 1 つの基底から派生している（まとめて捕まえられる）。"""
    for exc in (
        ocrus.ModelNotFoundError,
        ocrus.ModelError,
        ocrus.ImageError,
        ocrus.ConfigError,
    ):
        assert issubclass(exc, ocrus.OcrusError)


def test_missing_model_dir_raises_model_not_found(tmp_path: Path) -> None:
    """モデルが無いディレクトリを指したら、原因が分かる例外になる。"""
    with pytest.raises(ocrus.ModelNotFoundError) as excinfo:
        ocrus.OcrEngine(model_dir=tmp_path)
    message = str(excinfo.value)
    assert "rec.ocnn" in message
    assert str(tmp_path) in message


def test_invalid_mode_raises_config_error() -> None:
    """Mode の綴り間違いは ConfigError で弾く。"""
    with pytest.raises(ocrus.ConfigError):
        ocrus.OcrEngine(mode="fast")


def test_invalid_charset_raises_config_error() -> None:
    """文字集合の検証はモデルを読む前に行う。"""
    with pytest.raises(ocrus.ConfigError):
        ocrus.OcrEngine(charset="unknown")


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 3), (2, 3, 4)])
def test_raw_pixels_preserve_shape_and_noncontiguous_data(
    shape: tuple[int, ...],
) -> None:
    """非連続配列も行優先のバイト列として渡す。"""
    np = pytest.importorskip("numpy")
    array = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)[:, ::-1]
    data, width, height, channels = ocrus._as_raw_pixels(array)
    assert (width, height, channels) == (3, 2, shape[2] if len(shape) == 3 else 1)
    assert data == np.ascontiguousarray(array).tobytes()


def test_raw_pixels_convert_dtype_and_pil_mode() -> None:
    """uint8 以外の配列と、対応外の PIL モードを変換する。"""
    np = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")
    data, width, height, channels = ocrus._as_raw_pixels(
        np.array([[1, 2]], dtype=np.int16)
    )
    assert (data, width, height, channels) == (b"\x01\x02", 2, 1, 1)
    image = pil.new("P", (2, 3))
    assert ocrus._as_raw_pixels(image) == (image.convert("L").tobytes(), 2, 3, 1)


def test_raw_pixels_reject_unsupported_inputs() -> None:
    """画像として解釈できない型と次元数を弾く。"""
    np = pytest.importorskip("numpy")
    for invalid in [
        42,
        np.zeros((3,), dtype=np.uint8),
        np.zeros((1, 2, 3, 4), dtype=np.uint8),
    ]:
        with pytest.raises(TypeError):
            ocrus._as_raw_pixels(invalid)


@needs_model
@needs_sample
def test_recognize_path() -> None:
    """パスを渡すと結果の構造が揃って返る。"""
    engine = ocrus.OcrEngine()
    result = engine.recognize(SAMPLE)

    assert len(result.pages) == 1
    page = result.pages[0]
    assert (page.width, page.height) == (600, 200)
    assert page.lines, "行が 1 つも取れていない"

    line = page.lines[0]
    assert isinstance(line.text, str)
    assert 0.0 <= line.confidence <= 1.0
    assert line.bbox.as_tuple()[2] > 0
    assert result.full_text() == "\n".join(line.text for line in page.lines)


@needs_model
@needs_sample
def test_recognize_bytes_matches_path() -> None:
    """バイト列で渡してもパスと同じ結果になる。"""
    engine = ocrus.OcrEngine()
    from_path = engine.recognize(SAMPLE)
    from_bytes = engine.recognize(SAMPLE.read_bytes())
    assert from_bytes.full_text() == from_path.full_text()


@needs_model
@needs_sample
def test_to_json_round_trips() -> None:
    """to_json / to_dict は CLI の --format json と同じ形。"""
    engine = ocrus.OcrEngine()
    result = engine.recognize(SAMPLE)

    parsed = json.loads(result.to_json())
    assert parsed == result.to_dict()
    assert set(parsed) == {"pages"}
    assert set(parsed["pages"][0]) == {"width", "height", "lines"}
    assert set(parsed["pages"][0]["lines"][0]) == {"text", "bbox", "confidence", "ruby"}


@needs_model
@needs_sample
def test_recognize_numpy_array_matches_path() -> None:
    """Numpy 配列で渡してもパスと同じ結果になる。"""
    np = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")

    image = pil.open(SAMPLE).convert("L")
    array = np.asarray(image, dtype=np.uint8)

    engine = ocrus.OcrEngine()
    assert engine.recognize(array).full_text() == engine.recognize(SAMPLE).full_text()


@needs_model
def test_recognize_raw_rejects_wrong_size() -> None:
    """バイト数が形と合わなければ ImageError。黙って崩れた結果を返さない。"""
    engine = ocrus.OcrEngine()
    with pytest.raises(ocrus.ImageError):
        engine.recognize_raw(b"\x00" * 10, width=32, height=32, channels=1)


@needs_model
def test_recognize_rejects_unsupported_type() -> None:
    """画像として解釈できないものは TypeError。"""
    engine = ocrus.OcrEngine()
    with pytest.raises(TypeError):
        engine.recognize(42)
