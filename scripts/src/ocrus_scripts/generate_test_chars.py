#!/usr/bin/env python3
"""Generate character data files for OCR testing."""

from pathlib import Path

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "test_chars"
)


def _is_cjk_ideograph(c: str) -> bool:
    """Check if a character is a CJK ideograph.

    Returns:
        True if the character is in the CJK Unified Ideographs ranges.

    """
    cp = ord(c)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF


def _generate_halfwidth_alnum() -> str:
    """Generate half-width alphanumeric characters A-Z, a-z, 0-9.

    Returns:
        Concatenated string of all half-width alphanumeric characters.

    """
    chars = ""
    for c in range(ord("A"), ord("Z") + 1):
        chars += chr(c)
    for c in range(ord("a"), ord("z") + 1):
        chars += chr(c)
    for c in range(ord("0"), ord("9") + 1):
        chars += chr(c)
    return chars


def _generate_halfwidth_symbols() -> str:
    """Generate half-width symbol characters.

    Returns:
        Concatenated string of common half-width symbols.

    """
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    return symbols


def _generate_fullwidth_symbols() -> str:
    """Generate full-width symbols and Japanese punctuation.

    Returns:
        Concatenated string of full-width symbols and Japanese punctuation.

    """
    chars = ""
    # Full-width symbols U+FF01-U+FF0F, U+FF1A-U+FF20, U+FF3B-U+FF40, U+FF5B-U+FF5E
    for start, end in [
        (0xFF01, 0xFF0F),
        (0xFF1A, 0xFF20),
        (0xFF3B, 0xFF40),
        (0xFF5B, 0xFF5E),
    ]:
        chars += "".join(chr(c) for c in range(start, end + 1))
    # Japanese punctuation
    chars += "、。「」『』【】〒〜・ー々〇〈〉《》〔〕〖〗"
    # Additional common Japanese symbols
    chars += "…‥¥※♪†‡§¶"
    return chars


def _generate_fullwidth_alnum() -> str:
    """Generate full-width alphanumeric characters.

    Returns:
        Concatenated string of all full-width alphanumeric characters.

    """
    chars = ""
    for c in range(ord("\uff21"), ord("\uff3a") + 1):  # A-Z
        chars += chr(c)
    for c in range(ord("\uff41"), ord("\uff5a") + 1):  # a-z
        chars += chr(c)
    for c in range(ord("\uff10"), ord("\uff19") + 1):  # 0-9
        chars += chr(c)
    return chars


def _generate_hiragana() -> str:
    """Generate hiragana characters U+3041 to U+3093.

    Returns:
        Concatenated string of all hiragana characters.

    """
    return "".join(chr(c) for c in range(0x3041, 0x3093 + 1))


def _generate_katakana() -> str:
    """Generate katakana characters U+30A1 to U+30F6.

    Returns:
        Concatenated string of all katakana characters.

    """
    return "".join(chr(c) for c in range(0x30A1, 0x30F6 + 1))


def _generate_joyo_kanji() -> str:
    """Generate 常用漢字 (Joyo Kanji) 2010 revision - 2136 characters.

    Fetches the official list from Wikipedia's 常用漢字一覧 page.
    Falls back to reading the existing file if the fetch fails.

    Returns:
        Concatenated string of all 2136 Joyo Kanji.

    Raises:
        RuntimeError: If fetch fails and no fallback file exists.

    """
    import re
    import urllib.request

    existing = OUTPUT_DIR / "joyo_kanji.txt"

    try:
        url = (
            "https://ja.wikipedia.org/w/api.php"
            "?action=parse"
            "&page=%E5%B8%B8%E7%94%A8%E6%BC%A2%E5%AD%97"
            "%E4%B8%80%E8%A6%A7&prop=text&format=json"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
        import json

        data = json.loads(resp.read().decode("utf-8"))
        html = data["parse"]["text"]["*"]

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        joyo: dict[int, str] = {}
        for row in rows:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            if not cells:
                continue
            num_text = re.sub(r"<[^>]+>", "", cells[0]).strip()
            try:
                num = int(num_text)
            except ValueError:
                continue
            if not 1 <= num <= 2136:
                continue
            for c in re.findall(r'title="wikt:(.)"', row):
                if "\u4e00" <= c <= "\u9fff":
                    joyo[num] = c
                    break
            if num not in joyo:
                # Handle CJK Extension chars (e.g. 𠮟 U+20B9F -> 叱)
                for c in re.findall(r'title="wikt:(.)"', row):
                    if ord(c) >= 0x20000:
                        # Use the BMP alternative listed after it
                        alt = re.findall(r'title="wikt:(.)"', row)
                        for a in alt:
                            if "\u4e00" <= a <= "\u9fff":
                                joyo[num] = a
                                break
                        break

        if len(joyo) == 2136:
            return "".join(joyo[i] for i in range(1, 2137))

    except Exception:
        pass

    # Fallback: read existing file
    if existing.exists():
        content = existing.read_text(encoding="utf-8").strip()
        if len(content) >= 2100:
            return content

    msg = "Could not generate joyo kanji list (fetch failed, no fallback)"
    raise RuntimeError(msg)


def _generate_jis_x0208_kanji() -> tuple[str, str]:
    """Generate JIS X 0208 Level 1 and Level 2 kanji.

    Returns:
        Tuple of (level1_kanji, level2_kanji) strings.

    """
    level1_chars = []
    level2_chars = []

    for row in range(16, 84 + 1):
        for col in range(1, 94 + 1):
            b1 = row + 0xA0
            b2 = col + 0xA0
            try:
                c = bytes([b1, b2]).decode("euc_jp")
                if len(c) == 1 and _is_cjk_ideograph(c):
                    if row <= 47:
                        level1_chars.append(c)
                    else:
                        level2_chars.append(c)
            except (UnicodeDecodeError, ValueError):
                continue

    return "".join(level1_chars), "".join(level2_chars)


def _generate_jis_x0213_level3() -> str:
    """Generate JIS X 0213 Level 3 kanji (plane 1 chars not in JIS X 0208).

    Returns:
        Concatenated string of Level 3 kanji characters.

    """
    # First collect all JIS X 0208 chars
    x0208_chars = set()
    for row in range(1, 94 + 1):
        for col in range(1, 94 + 1):
            b1 = row + 0xA0
            b2 = col + 0xA0
            try:
                c = bytes([b1, b2]).decode("euc_jp")
                if len(c) == 1:
                    x0208_chars.add(c)
            except (UnicodeDecodeError, ValueError):
                continue

    # Now iterate plane 1 in euc_jis_2004
    level3_chars = []
    for row in range(1, 94 + 1):
        for col in range(1, 94 + 1):
            b1 = row + 0xA0
            b2 = col + 0xA0
            try:
                c = bytes([b1, b2]).decode("euc_jis_2004")
                if len(c) == 1 and _is_cjk_ideograph(c) and c not in x0208_chars:
                    level3_chars.append(c)
            except (UnicodeDecodeError, ValueError):
                continue

    return "".join(level3_chars)


def _generate_jis_x0213_level4() -> str:
    """Generate JIS X 0213 Level 4 kanji (plane 2, 0x8F prefix).

    Returns:
        Concatenated string of Level 4 kanji characters.

    """
    level4_chars = []
    for row in range(1, 94 + 1):
        for col in range(1, 94 + 1):
            b1 = row + 0xA0
            b2 = col + 0xA0
            try:
                c = bytes([0x8F, b1, b2]).decode("euc_jis_2004")
                if len(c) == 1 and _is_cjk_ideograph(c):
                    level4_chars.append(c)
            except (UnicodeDecodeError, ValueError):
                continue

    return "".join(level4_chars)


def main() -> None:
    """Generate all test character files and print statistics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {}

    files["halfwidth_alnum.txt"] = _generate_halfwidth_alnum()
    files["halfwidth_symbols.txt"] = _generate_halfwidth_symbols()
    files["fullwidth_alnum.txt"] = _generate_fullwidth_alnum()
    files["fullwidth_symbols.txt"] = _generate_fullwidth_symbols()
    files["hiragana.txt"] = _generate_hiragana()
    files["katakana.txt"] = _generate_katakana()

    files["joyo_kanji.txt"] = _generate_joyo_kanji()

    level1, level2 = _generate_jis_x0208_kanji()
    files["jis_level1.txt"] = level1
    files["jis_level2.txt"] = level2
    files["jis_level3.txt"] = _generate_jis_x0213_level3()
    files["jis_level4.txt"] = _generate_jis_x0213_level4()

    print("=== OCR Test Character Generation ===\n")
    for filename, chars in files.items():
        path = OUTPUT_DIR / filename
        path.write_text(chars, encoding="utf-8")
        print(f"{filename}: {len(chars)} chars")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
