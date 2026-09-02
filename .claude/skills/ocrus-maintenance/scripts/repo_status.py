"""ocrus の状態を 1 画面で確認する。

メンテナンスの最初と、ビルドや依存更新のあとに回す。見たいのは次の 5 つ。

- 未コミットの変更があるか（depup が Cargo.toml / pyproject.toml を書き換える前に見る）
- モデルが揃っているか（rec.ocnn と dict.txt が無いと認識も精度テストも動かない）
- ビルド成果物が Rust ソースより新しいか（古ければ再ビルドが要る）
- 精度の直近値（logs/char_accuracy_*.log の最後の Overall Accuracy）
- 道具が揃っているか

使い方:
    python .claude/skills/ocrus-maintenance/scripts/repo_status.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]

# ビルド成果物より新しいと再ビルドが要るソースの置き場所。
# target/ を見ないのは、成果物自身の更新で「ソースが新しい」と誤判定しないため。
RUST_SOURCE_DIRS = ("crates", "benches")
RUST_SOURCE_FILES = ("Cargo.toml", "Cargo.lock")

MODEL_FILES = (
    ("rec.ocnn", True, "認識モデル本体。無いと recognize も精度テストも動かない"),
    ("dict.txt", True, "文字辞書 18,383 行。無いと CTC デコードができない"),
    ("rec.onnx", False, "変換元。.ocnn があれば実行には不要"),
)

STEPS = (
    "step1",
    "step2",
    "step3_joyo",
    "step3_jis1",
    "step3_jis2",
    "step3_jis3",
    "step3_jis4",
)

CATEGORY_RE = re.compile(
    r"^\s*(?P<category>[a-z0-9_]+)\s+(?:FP32=)?(?P<correct>\d+)/(?P<total>\d+)\s+\((?P<pct>[\d.]+)%\)"
)


def run(args: list[str], cwd: Path = ROOT) -> str:
    """コマンドを実行して出力を返す。

    Args:
        args: 実行するコマンドと引数。
        cwd: 実行ディレクトリ。

    Returns:
        標準出力（空なら標準エラー）。実行できなければその旨の文字列。

    """
    try:
        out = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"<実行できない: {exc}>"
    return (out.stdout or out.stderr).strip()


def fmt_age(mtime: float) -> str:
    """更新時刻を「N 分前」の形にする。

    Args:
        mtime: エポック秒。

    Returns:
        人が読める経過時間。0 以下なら「なし」。

    """
    if mtime <= 0:
        return "なし"
    delta = time.time() - mtime
    if delta < 3600:
        return f"{delta / 60:.0f} 分前"
    if delta < 86400:
        return f"{delta / 3600:.1f} 時間前"
    return f"{delta / 86400:.1f} 日前"


def fmt_size(n: int) -> str:
    """バイト数を人が読める単位にする。

    Args:
        n: バイト数。

    Returns:
        MB / KB / B のいずれかで表した文字列。

    """
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} KB"
    return f"{n} B"


def newest_rust_mtime() -> float:
    """Rust ソースの中で最も新しい更新時刻を返す。

    Returns:
        エポック秒。ソースが 1 つも無ければ 0.0。

    """
    newest = 0.0
    for name in RUST_SOURCE_FILES:
        path = ROOT / name
        if path.exists():
            newest = max(newest, path.stat().st_mtime)
    for dirname in RUST_SOURCE_DIRS:
        base = ROOT / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*.rs"):
            if "target" in path.parts:
                continue
            newest = max(newest, path.stat().st_mtime)
    return newest


def model_dir() -> Path:
    """モデルディレクトリを解決する。

    Returns:
        OCRUS_MODEL_DIR があればそれ、無ければ ~/.ocrus/models。

    """
    override = os.environ.get("OCRUS_MODEL_DIR")
    if override:
        return Path(override)
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or "."
    return Path(home) / ".ocrus" / "models"


def report_git() -> None:
    """HEAD と未コミットの変更を出す。"""
    print("\n## リポジトリ")
    print(f"  HEAD      : {run(['git', 'log', '--oneline', '-1'])}")
    print(f"  ブランチ  : {run(['git', 'branch', '--show-current'])}")

    dirty = [
        line
        for line in run(["git", "status", "--porcelain"]).splitlines()
        if line.strip()
    ]
    if dirty:
        print(f"  未コミット: {len(dirty)} 件  ← depup より先に検証を通して確定させる")
        for line in dirty[:12]:
            print(f"              {line}")
        if len(dirty) > 12:
            print(f"              ... 他 {len(dirty) - 12} 件")
    else:
        print("  未コミット: なし")


def report_models() -> None:
    """モデルファイルの有無とサイズを出す。"""
    mdir = model_dir()
    origin = "OCRUS_MODEL_DIR" if os.environ.get("OCRUS_MODEL_DIR") else "既定"
    print(f"\n## モデル  ({mdir}, {origin})")
    if not mdir.is_dir():
        print("  !! ディレクトリが無い。python models/download.py で取得する")
        return
    for name, required, note in MODEL_FILES:
        path = mdir / name
        if path.exists():
            st = path.stat()
            print(f"  {name:<10}: {fmt_size(st.st_size):>9}  {fmt_age(st.st_mtime)}")
        elif required:
            print(f"  {name:<10}: なし  ← {note}。python models/download.py")
        else:
            print(f"  {name:<10}: なし  （{note}）")


def report_build() -> None:
    """Release ビルドがソースより新しいかを出す。"""
    src_mtime = newest_rust_mtime()
    print("\n## ビルド")
    print(f"  Rust ソース: 最終更新 {fmt_age(src_mtime)}")
    release = ROOT / "target" / "release"
    binary = next(
        (p for p in (release / "ocrus.exe", release / "ocrus") if p.exists()),
        None,
    )
    if binary is None:
        print("  release   : 未ビルド  ← cargo build --release")
        return
    st = binary.stat()
    stale = (
        " ← ソースより古い。cargo build --release が要る"
        if st.st_mtime < src_mtime
        else ""
    )
    print(f"  release   : {binary.name}  {fmt_age(st.st_mtime)}{stale}")


def report_test_assets() -> None:
    """精度テストが要る素材（test_images / test_chars / failures）を出す。"""
    print("\n## 精度テストの前提")
    images = ROOT / "test_images"
    if images.is_dir():
        fonts = sorted(p.name for p in images.iterdir() if p.is_dir())
        more = " ..." if len(fonts) > 4 else ""
        print(f"  test_images: {len(fonts)} フォント  ({', '.join(fonts[:4])}{more})")
    else:
        print(
            "  test_images: なし  ← cargo test -p ocrus-cli "
            "--test generate_test_images -- --ignored --nocapture"
        )

    chars = ROOT / "data" / "test_chars"
    if chars.is_dir():
        cats = sorted(p.stem for p in chars.glob("*.txt"))
        print(f"  test_chars : {len(cats)} カテゴリ")
    else:
        print("  test_chars : なし  ← data/test_chars/ が要る")

    results = ROOT / "test_results"
    if results.is_dir():
        for path in sorted(results.glob("failures_*.json")):
            st = path.stat()
            print(
                f"  {path.name:<22}: {fmt_size(st.st_size):>9}  {fmt_age(st.st_mtime)}"
            )


def parse_overall(path: Path) -> list[tuple[str, str, float]]:
    """ログ末尾の Overall Accuracy ブロックを読む。

    Args:
        path: char_accuracy のログファイル。

    Returns:
        (category, "correct/total", pct) の一覧。ブロックが無ければ空。

    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    idx = text.rfind("=== Overall Accuracy")
    if idx < 0:
        return []
    rows: list[tuple[str, str, float]] = []
    for line in text[idx:].splitlines()[1:]:
        body = line.split("[INFO]", 1)[-1]
        m = CATEGORY_RE.match(body)
        if not m:
            if "===" in line:
                break
            continue
        rows.append((m["category"], f"{m['correct']}/{m['total']}", float(m["pct"])))
    return rows


def report_accuracy() -> None:
    """直近の精度測定値をログから拾って出す。"""
    print("\n## 直近の精度（logs/char_accuracy_*.log の最後の Overall Accuracy）")
    logs = ROOT / "logs"
    if not logs.is_dir():
        print("  logs/ が無い")
        return
    found = False
    for step in STEPS:
        candidates = sorted(
            logs.glob(f"char_accuracy_{step}_*.log"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            continue
        latest = candidates[-1]
        rows = parse_overall(latest)
        if not rows:
            continue
        found = True
        print(f"  {step}  ({latest.name}, {fmt_age(latest.stat().st_mtime)})")
        for category, ratio, pct in rows:
            print(f"    {category:<20} {ratio:>10}  {pct:5.1f}%")
    if not found:
        print(
            "  まだ測定していない"
            "（精度テストはユーザーに実行を依頼する。SKILL.md 手順 5）"
        )


def report_python() -> None:
    """Python 側（scripts の venv、バインディングの wheel）の状態を出す。"""
    print("\n## Python 側")
    venv = ROOT / "scripts" / ".venv"
    state = (
        "あり" if venv.is_dir() else "なし  ← 学習・変換を回す回だけ uv sync -C scripts"
    )
    print(f"  scripts/.venv : {state}")

    if not (ROOT / "crates" / "ocrus-python").is_dir():
        return

    wheels = sorted(
        (ROOT / "target" / "wheels").glob("ocrus-*.whl"),
        key=lambda p: p.stat().st_mtime,
    )
    if wheels:
        latest = wheels[-1]
        src = newest_rust_mtime()
        stale = " ← Rust ソースより古い" if latest.stat().st_mtime < src else ""
        print(
            f"  wheel         : {latest.name}  {fmt_age(latest.stat().st_mtime)}{stale}"
        )
    else:
        print(
            "  wheel         : 未ビルド  ← cd python && "
            "uvx maturin build --release --out ../target/wheels"
        )


def report_tools() -> None:
    """道具のバージョンを出す。"""
    print("\n## 道具")
    for label, args in (
        ("depup", ["depup", "--version"]),
        ("cargo", ["cargo", "--version"]),
        ("rustc", ["rustc", "--version"]),
        ("uv", ["uv", "--version"]),
        ("maturin", ["maturin", "--version"]),
    ):
        out = run(args)
        first = out.splitlines()[0] if out else "見つからない"
        print(f"  {label:<8}: {first}")


def main() -> None:
    """全セクションを順に出す。"""
    print("=" * 78)
    print(f"ocrus の状態  ({ROOT})")
    print("=" * 78)
    report_git()
    report_models()
    report_build()
    report_test_assets()
    report_accuracy()
    report_python()
    report_tools()
    print("\n" + "=" * 78)
    print("未コミットの変更が残っていたら、depup より先に検証を通して確定させること。")
    print("=" * 78)


if __name__ == "__main__":
    main()
