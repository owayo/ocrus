"""精度ログ 2 本を突き合わせて、カテゴリごとの増減を出す。

`cargo test --test char_accuracy` は実行のたびに
logs/char_accuracy_<step>_<epoch>.log を残す。同じ step・同じ test_images に対しては
決定的に動く（同じ画像・同じモデル・同じデコード経路）ので、2 本の差はそのまま
コードの変更による差になる。
逆に言うと **分母 (correct/total の total) が違う 2 本を比べても意味がない** ので、
分母が食い違ったら警告を出す。フォントを足した / カテゴリのファイルを触った回は
比較ではなく測り直しが要る。

使い方:
    python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py step1
        step1 の最新 2 本を比較する（1 本前 → 最新）

    python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py step1 \
        --base logs/char_accuracy_step1_1773072063.log \
        --new  logs/char_accuracy_step1_1773091826.log

    python .claude/skills/ocrus-maintenance/scripts/compare_accuracy.py --list
        step ごとのログ一覧（新しい順）
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
LOGS = ROOT / "logs"

CATEGORY_RE = re.compile(
    r"^\s*(?P<category>[a-z0-9_]+)\s+(?:FP32=)?(?P<correct>\d+)/(?P<total>\d+)\s+\((?P<pct>[\d.]+)%\)"
)


def parse_overall(path: Path) -> dict[str, tuple[int, int]]:
    """ログ末尾の Overall Accuracy ブロックを読む。

    途中経過の行も同じ形をしているので、最後の `=== Overall Accuracy` 以降だけを読む。

    Args:
        path: char_accuracy のログファイル。

    Returns:
        {category: (correct, total)}。中断された（Ctrl+C）ログには
        このブロックが無く、空の dict を返す。

    """
    text = path.read_text(encoding="utf-8", errors="replace")
    idx = text.rfind("=== Overall Accuracy")
    if idx < 0:
        return {}
    rows: dict[str, tuple[int, int]] = {}
    for line in text[idx:].splitlines()[1:]:
        body = line.split("[INFO]", 1)[-1]
        m = CATEGORY_RE.match(body)
        if not m:
            if "===" in line:
                break
            continue
        rows[m["category"]] = (int(m["correct"]), int(m["total"]))
    return rows


def logs_for(step: str) -> list[Path]:
    """その step のログを古い順に返す。

    Args:
        step: step1 / step2 / step3_joyo などのラベル。

    Returns:
        更新時刻の古い順に並べたログのパス。

    """
    return sorted(
        LOGS.glob(f"char_accuracy_{step}_*.log"), key=lambda p: p.stat().st_mtime
    )


def stamp(path: Path) -> str:
    """ログの更新時刻を人が読める形にする。

    Args:
        path: 対象のファイル。

    Returns:
        `YYYY-MM-DD HH:MM` 形式の文字列。

    """
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def pct(correct: int, total: int) -> float:
    """正解率をパーセントで返す。

    Args:
        correct: 正解数。
        total: 試行数。

    Returns:
        パーセント。total が 0 なら 0.0。

    """
    return correct / total * 100.0 if total else 0.0


def list_logs() -> int:
    """Step ごとのログ一覧を新しい順に出す。

    Returns:
        終了コード。logs/ が無ければ 1。

    """
    if not LOGS.is_dir():
        print("logs/ が無い")
        return 1
    steps: dict[str, list[Path]] = {}
    for path in LOGS.glob("char_accuracy_*.log"):
        m = re.match(r"char_accuracy_(?P<step>.+)_\d+\.log$", path.name)
        if m:
            steps.setdefault(m["step"], []).append(path)
    for step in sorted(steps):
        print(f"\n## {step}")
        for path in sorted(steps[step], key=lambda p: p.stat().st_mtime, reverse=True):
            rows = parse_overall(path)
            summary = (
                "  ".join(f"{c}={pct(*v):.1f}%" for c, v in rows.items())
                if rows
                else "(Overall Accuracy 無し — 中断されたログ)"
            )
            print(f"  {stamp(path)}  {path.name}")
            print(f"      {summary}")
    return 0


def compare(base: Path, new: Path) -> int:
    """2 本のログをカテゴリごとに突き合わせる。

    Args:
        base: 比較元のログ。
        new: 比較先のログ。

    Returns:
        終了コード。分母が食い違う、または Overall Accuracy が無ければ 1。

    """
    base_rows = parse_overall(base)
    new_rows = parse_overall(new)
    if not base_rows or not new_rows:
        missing = base if not base_rows else new
        print(
            f"!! {missing.name} に Overall Accuracy が無い。"
            "中断されたログの可能性がある"
        )
        return 1

    print(f"base: {base.name}  ({stamp(base)})")
    print(f"new : {new.name}  ({stamp(new)})")
    print()
    print(f"{'category':<22}{'base':>18}{'new':>18}{'diff':>10}")
    print("-" * 68)

    mismatched: list[str] = []
    b_correct = b_total = n_correct = n_total = 0

    for category in sorted(set(base_rows) | set(new_rows)):
        b = base_rows.get(category)
        n = new_rows.get(category)
        if b is None or n is None:
            side = "base" if b is None else "new"
            print(f"{category:<22}{'(' + side + ' に無い)':>46}")
            continue
        if b[1] != n[1]:
            mismatched.append(category)
        bp, np_ = pct(*b), pct(*n)
        diff = np_ - bp
        mark = "" if abs(diff) < 0.05 else ("  ↑" if diff > 0 else "  ↓")
        print(
            f"{category:<22}{f'{b[0]}/{b[1]} ({bp:.1f}%)':>18}"
            f"{f'{n[0]}/{n[1]} ({np_:.1f}%)':>18}{f'{diff:+.1f}pt':>10}{mark}"
        )
        b_correct += b[0]
        b_total += b[1]
        n_correct += n[0]
        n_total += n[1]

    if b_total and n_total:
        bp, np_ = pct(b_correct, b_total), pct(n_correct, n_total)
        print("-" * 68)
        print(
            f"{'合計':<21}{f'{b_correct}/{b_total} ({bp:.1f}%)':>18}"
            f"{f'{n_correct}/{n_total} ({np_:.1f}%)':>18}{f'{np_ - bp:+.1f}pt':>10}"
        )

    if mismatched:
        print()
        print("!! 分母が違うカテゴリがある: " + ", ".join(mismatched))
        print(
            "   test_images のフォントか data/test_chars の中身が変わっている。"
            "この 2 本は比較にならないので、同じ条件で測り直す。"
        )
        return 1

    print()
    print(
        "同じ step・同じ test_images なら char_accuracy は決定的に動くので、"
        "この差はそのままコードの差。"
    )
    print("0.1pt 未満の差しか出ていないなら「変わっていない」と読む。")
    return 0


def main() -> int:
    """コマンドライン引数を解いて比較か一覧を実行する。

    Returns:
        終了コード。

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", nargs="?", help="step1 / step2 / step3_joyo など")
    parser.add_argument("--base", type=Path, help="比較元のログ（既定: 最新の 1 本前）")
    parser.add_argument("--new", type=Path, help="比較先のログ（既定: 最新）")
    parser.add_argument("--list", action="store_true", help="ログ一覧を出して終わる")
    args = parser.parse_args()

    if args.list or not args.step:
        return list_logs()

    if args.base and args.new:
        return compare(args.base, args.new)

    candidates = logs_for(args.step)
    if len(candidates) < 2:
        print(f"{args.step} のログが {len(candidates)} 本しかない。比較には 2 本要る")
        return 1
    return compare(candidates[-2], candidates[-1])


if __name__ == "__main__":
    sys.exit(main())
