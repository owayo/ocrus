"""失敗リスト (test_results/failures_*.json) を読んで、誤りの傾向を出す。

精度の数字だけでは次の一手が決まらない。同じ 40% でも「空出力ばかり」と
「似た文字への安定誤認識ばかり」では効く手が違う（todo.md の記録では、
空出力寄りの記号カテゴリには TLA が効き、安定誤認識のかなには効かなかった）。
このスクリプトはその切り分けをする。

使い方:
    python .claude/skills/ocrus-model-improvement/scripts/failure_report.py step1
        test_results/failures_step1.json の傾向を出す

    python .claude/skills/ocrus-model-improvement/scripts/failure_report.py step1 \
        --diff test_results/failures_step1.base.json
        変更前に控えておいた失敗リストと突き合わせ、
        直った文字と新しく壊れた文字を出す

改善を試す前に必ず控えを取っておくこと。失敗リストは実行のたびに上書きされる:

    cp test_results/failures_step1.json test_results/failures_step1.base.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "test_results"

SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")


def load(path: Path) -> list[dict[str, str]]:
    """失敗リストを読む。

    Args:
        path: failures_*.json のパス。

    Returns:
        {character, category, font_name, expected, recognized} の一覧。

    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"!! 読めない: {path} ({exc})")
        return []
    return data if isinstance(data, list) else []


def key(row: dict[str, str]) -> tuple[str, str, str]:
    """1 件の失敗を同定するキー。

    Args:
        row: 失敗 1 件。

    Returns:
        (フォント, カテゴリ, 文字) の組。

    """
    return (row.get("font_name", ""), row.get("category", ""), row.get("expected", ""))


def describe(ch: str) -> str:
    """文字を「あ (U+3042)」の形にする。

    Args:
        ch: 対象の文字。空文字なら空出力を意味する。

    Returns:
        表示用の文字列。

    """
    if not ch:
        return "(空出力)"
    return f"{ch} (U+{ord(ch[0]):04X})"


def report(rows: list[dict[str, str]], label: str) -> None:
    """傾向をまとめて出す。

    Args:
        rows: 失敗の一覧。
        label: 見出しに出す step 名。

    """
    total = len(rows)
    if total == 0:
        print(f"{label}: 失敗 0 件")
        return

    empty = [r for r in rows if not r.get("recognized")]
    print(f"\n## {label}  失敗 {total} 件")
    print(f"  空出力          : {len(empty)} 件 ({len(empty) / total * 100:.1f}%)")
    small = [r for r in rows if r.get("expected", "") in SMALL_KANA]
    if small:
        print(f"  小書きかな      : {len(small)} 件 ({len(small) / total * 100:.1f}%)")

    print("\n## カテゴリ別")
    by_cat = Counter(r.get("category", "?") for r in rows)
    empty_by_cat = Counter(r.get("category", "?") for r in empty)
    for cat, n in by_cat.most_common():
        share = n / total * 100
        print(f"  {cat:<20} {n:>5} 件 ({share:4.1f}%)   空出力 {empty_by_cat[cat]:>4}")

    print("\n## フォント別")
    for font, n in Counter(r.get("font_name", "?") for r in rows).most_common():
        print(f"  {font:<28} {n:>5} 件")

    print("\n## 混同ペア上位 20（期待 → 認識）")
    pairs = Counter(
        (r.get("expected", ""), r.get("recognized", ""))
        for r in rows
        if r.get("recognized")
    )
    for (exp, rec), n in pairs.most_common(20):
        same_shape = (
            "  ← 字形が近い"
            if unicodedata.normalize("NFKC", exp).lower()
            == unicodedata.normalize("NFKC", rec).lower()
            else ""
        )
        print(f"  {describe(exp):<16} → {describe(rec):<16} {n:>4} 件{same_shape}")

    print("\n## 読み方")
    empty_share = len(empty) / total
    if empty_share > 0.2:
        print(
            "  空出力が多い。モデルが「何も無い」と判定している類。"
            "時系列 logit の集約 (TLA) や前処理バリアントが効きやすい層。"
        )
    else:
        print(
            "  空出力は少なく、似た文字への安定誤認識が主。"
            "デコード側の小細工では動かない。字形の差を残す前処理か、"
            "文脈・頻度事前分布のような別情報が要る。"
        )
    print("  todo.md の「試して失敗した手法」を必ず先に読むこと。")


def report_diff(base: list[dict[str, str]], new: list[dict[str, str]]) -> None:
    """2 つの失敗リストを突き合わせる。

    Args:
        base: 変更前の失敗一覧。
        new: 変更後の失敗一覧。

    """
    base_keys = {key(r) for r in base}
    new_keys = {key(r) for r in new}
    fixed = base_keys - new_keys
    broken = new_keys - base_keys

    print(f"\n## 差分  base {len(base_keys)} 件 → new {len(new_keys)} 件")
    print(f"  直った          : {len(fixed)} 件")
    print(f"  新しく壊れた    : {len(broken)} 件")
    print(f"  差し引き        : {len(new_keys) - len(base_keys):+d} 件")

    def dump(
        title: str, keys: set[tuple[str, str, str]], rows: list[dict[str, str]]
    ) -> None:
        if not keys:
            return
        print(f"\n### {title}（上位 30）")
        index = {key(r): r for r in rows}
        by_cat: Counter[str] = Counter(k[1] for k in keys)
        for cat, n in by_cat.most_common():
            print(f"  {cat:<20} {n:>4} 件")
        print("  例:")
        for k in sorted(keys)[:30]:
            row = index.get(k, {})
            rec = row.get("recognized", "")
            print(f"    {k[1]:<20} {describe(k[2]):<16} → {describe(rec)}  [{k[0]}]")

    dump("直った", fixed, base)
    dump("新しく壊れた", broken, new)

    if broken and not fixed:
        print("\n  一方的に悪化している。撤回して原因を切り分ける。")
    elif fixed and broken:
        print(
            "\n  直りと壊れが両方ある。合計だけ見て採用を決めない。"
            "壊れた側が本命のカテゴリなら差し引きプラスでも見送る。"
        )


def resolve(arg: str) -> Path:
    """Step 名かパスを failures_*.json のパスに解決する。

    Args:
        arg: step1 のようなラベル、またはファイルパス。

    Returns:
        失敗リストのパス。

    """
    path = Path(arg)
    if path.exists():
        return path
    return RESULTS / f"failures_{arg}.json"


def main() -> int:
    """コマンドライン引数を解いて集計する。

    Returns:
        終了コード。

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", help="step1 / step2、または failures json のパス")
    parser.add_argument("--diff", help="比較元の failures json（変更前に控えたもの）")
    args = parser.parse_args()

    target = resolve(args.step)
    if not target.exists():
        print(f"!! {target} が無い。精度テストを回すと作られる")
        return 1

    rows = load(target)
    report(rows, target.stem)

    if args.diff:
        base_path = resolve(args.diff)
        if not base_path.exists():
            print(f"!! {base_path} が無い")
            return 1
        report_diff(load(base_path), rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
