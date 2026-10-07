"""Generate a fixed UI corpus and measure the production recognize_batch example.

Rendering needs Pillow; evaluation only uses the Python standard library. Images
and model hashes are saved so comparisons can verify identical inputs and weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

DEV_TEXTS = (
    "ファイルを保存する",
    "検索結果：128件",
    "設定を変更しました",
    "バージョンv2.4.1",
    "読み込み中です…",
    "更新日時：2026/10/07",
    "次のページへ進む",
    "エラー：接続できません",
)
TEST_TEXTS = (
    "名前を付けて保存",
    "ダウンロード完了",
    "残り時間：32秒",
    "表示倍率：125%",
    "お気に入りに追加",
    "パス：C:/Users/test",
    "ユーザーID：abc123",
    "アップデートを確認",
)


def digest(path: Path) -> str:
    """Hash a file.

    Returns:
        The SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(args: argparse.Namespace) -> None:
    """Save the selected corpus, rendered at native resolution, and its manifest."""
    from PIL import Image, ImageDraw, ImageFont, __version__

    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for font_path in args.fonts:
        for size in args.sizes:
            font = ImageFont.truetype(str(font_path), size)
            for index, text in enumerate(
                DEV_TEXTS if args.split == "dev" else TEST_TEXTS
            ):
                left, top, right, bottom = font.getbbox(text)
                ink_width, ink_height = right - left, bottom - top
                for theme, bg, fg in (("light", 255, 0), ("dark", 24, 230)):
                    for margin in ("compact", "wide"):
                        pad = max(3, size // 4)
                        width = ink_width + 2 * pad
                        if margin == "wide":
                            width = max(width, 1200)
                        image = Image.new("L", (width, ink_height + 2 * pad), bg)
                        ImageDraw.Draw(image).text(
                            (pad - left, pad - top), text, font=font, fill=fg
                        )
                        path = args.output / f"{len(records):05}.png"
                        image.save(path)
                        records.append(
                            {
                                "image": str(path.resolve()),
                                "sha256": digest(path),
                                "expected": text,
                                "font": font_path.stem,
                                "size": size,
                                "theme": theme,
                                "margin": margin,
                                "text_index": index,
                            }
                        )
    manifest = {
        "split": args.split,
        "renderer": f"Pillow {__version__}",
        "fonts": {str(p.resolve()): digest(p) for p in args.fonts},
        "records": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rendered {len(records)} images in {args.output}", flush=True)


def errors(expected: str, got: str) -> tuple[int, int, int]:
    """Return substitution, deletion and insertion counts for Levenshtein alignment.

    Ties prefer substitution, then deletion, then insertion.

    Returns:
        Substitution, deletion and insertion counts, in that order.
    """
    previous = [(i, 0, 0, i) for i in range(len(got) + 1)]
    for i, char in enumerate(expected, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(got, 1):
            if char == other:
                current.append(previous[j - 1])
            else:
                cost, sub, delete, insert = previous[j - 1]
                substitution = (cost + 1, sub + 1, delete, insert)
                cost, sub, delete, insert = previous[j]
                deletion = (cost + 1, sub, delete + 1, insert)
                cost, sub, delete, insert = current[j - 1]
                insertion = (cost + 1, sub, delete, insert + 1)
                current.append(
                    min((substitution, deletion, insertion), key=lambda item: item[0])
                )
        previous = current
    return previous[-1][1:]


def evaluate(args: argparse.Namespace) -> None:
    """Run the production engine and save strict character and exact-match metrics.

    Raises:
        ValueError: The corpus is empty or changed, or result counts do not match.
    """
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if not records:
        raise ValueError("The corpus is empty")
    for record in records:
        if digest(Path(record["image"])) != record["sha256"]:
            raise ValueError(f"Image changed: {record['image']}")
    totals = defaultdict(
        lambda: {
            "images": 0,
            "characters": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "exact": 0,
        }
    )
    predictions = []
    start = time.monotonic()
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        result = subprocess.run(
            [str(args.binary.resolve()), str(args.models.resolve()), str(args.threads)],
            input=json.dumps([r["image"] for r in batch]),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        outputs = json.loads(result.stdout)
        if len(outputs) != len(batch):
            raise ValueError("Recognizer returned the wrong number of images")
        for record, output in zip(batch, outputs, strict=True):
            got = "\n".join(
                line["text"] for page in output["pages"] for line in page["lines"]
            )
            sub, delete, insert = errors(record["expected"], got)
            predictions.append(
                {
                    **record,
                    "recognized": got,
                    "substitutions": sub,
                    "deletions": delete,
                    "insertions": insert,
                }
            )
            keys = ["all"] + [
                f"{key}={record[key]}"
                for key in ("font", "size", "theme", "margin", "category")
                if key in record
            ]
            keys.append(f"{record['theme']}/{record['margin']}")
            for key in keys:
                count = totals[key]
                count["images"] += 1
                count["characters"] += len(record["expected"])
                count["substitutions"] += sub
                count["deletions"] += delete
                count["insertions"] += insert
                count["exact"] += got == record["expected"]
        print(
            f"Evaluated {min(offset + args.batch_size, len(records))}/{len(records)} images",
            flush=True,
        )
    for count in totals.values():
        count["errors"] = (
            count["substitutions"] + count["deletions"] + count["insertions"]
        )
        count["cer"] = count["errors"] / count["characters"]
    report = {
        "completed": True,
        "manifest_sha256": digest(args.manifest),
        "model_sha256": digest(args.models / "rec.ocnn"),
        "dictionary_sha256": digest(args.models / "dict.txt"),
        "binary_sha256": digest(args.binary),
        "seconds": time.monotonic() - start,
        "totals": dict(totals),
        "predictions": predictions,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """Parse the command line and render or evaluate a fixed corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("render")
    generate.add_argument("--fonts", type=Path, nargs="+", required=True)
    generate.add_argument("--sizes", type=int, nargs="+", default=[12, 16, 24, 32])
    generate.add_argument("--split", choices=["dev", "test"], required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(run=render)
    measure = commands.add_parser("evaluate")
    measure.add_argument("--manifest", type=Path, required=True)
    measure.add_argument("--binary", type=Path, required=True)
    measure.add_argument("--models", type=Path, required=True)
    measure.add_argument("--threads", type=int, default=4)
    measure.add_argument("--batch-size", type=int, default=16)
    measure.add_argument("--output", type=Path, required=True)
    measure.set_defaults(run=evaluate)
    args = parser.parse_args()
    if (
        any(value <= 0 for value in getattr(args, "sizes", [1]))
        or getattr(args, "batch_size", 1) <= 0
        or getattr(args, "threads", 1) <= 0
    ):
        parser.error("sizes, threads and batch size must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
