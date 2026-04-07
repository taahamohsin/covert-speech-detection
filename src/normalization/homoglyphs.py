"""
Homoglyph normalizer: maps Unicode lookalike characters to ASCII equivalents.
Covers Cyrillic, Greek, fullwidth Latin, and mathematical alphanumeric symbols.
Must run before leetspeak decoding so Cyrillic chars become ASCII first.
"""

import unicodedata

import pandas as pd

# Codepoint-level map: Unicode char → ASCII replacement
HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic lookalikes
    "\u0430": "a",  # а → a
    "\u0435": "e",  # е → e
    "\u043e": "o",  # о → o
    "\u0440": "p",  # р → p
    "\u0441": "c",  # с → c
    "\u0445": "x",  # х → x
    "\u0456": "i",  # і → i
    "\u0458": "j",  # ј → j
    "\u0455": "s",  # ѕ → s
    "\u0410": "A",  # А → A
    "\u0412": "B",  # В → B
    "\u0415": "E",  # Е → E
    "\u041a": "K",  # К → K
    "\u041c": "M",  # М → M
    "\u041d": "H",  # Н → H
    "\u041e": "O",  # О → O
    "\u0420": "P",  # Р → P
    "\u0421": "C",  # С → C
    "\u0422": "T",  # Т → T
    "\u0425": "X",  # Х → X
    # Greek lookalikes (uppercase)
    "\u0391": "A",  # Α → A
    "\u0392": "B",  # Β → B
    "\u0395": "E",  # Ε → E
    "\u0396": "Z",  # Ζ → Z
    "\u0397": "H",  # Η → H
    "\u0399": "I",  # Ι → I
    "\u039a": "K",  # Κ → K
    "\u039c": "M",  # Μ → M
    "\u039d": "N",  # Ν → N
    "\u039f": "O",  # Ο → O
    "\u03a1": "P",  # Ρ → P
    "\u03a4": "T",  # Τ → T
    "\u03a5": "Y",  # Υ → Y
    "\u03a7": "X",  # Χ → X
    # Greek lookalikes (lowercase)
    "\u03bf": "o",  # ο → o
    "\u03c1": "p",  # ρ → p
    "\u03b9": "i",  # ι → i
    "\u03bd": "v",  # ν → v
    # Fullwidth Latin (common evasion)
    "\uff21": "A", "\uff22": "B", "\uff23": "C", "\uff24": "D",
    "\uff25": "E", "\uff26": "F", "\uff27": "G", "\uff28": "H",
    "\uff29": "I", "\uff2a": "J", "\uff2b": "K", "\uff2c": "L",
    "\uff2d": "M", "\uff2e": "N", "\uff2f": "O", "\uff30": "P",
    "\uff31": "Q", "\uff32": "R", "\uff33": "S", "\uff34": "T",
    "\uff35": "U", "\uff36": "V", "\uff37": "W", "\uff38": "X",
    "\uff39": "Y", "\uff3a": "Z",
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
    # Mathematical alphanumeric symbols (bold, italic, etc.)
    "\u210a": "g",  # ℊ
    "\u210b": "H",  # ℋ
    "\u210c": "H",  # ℌ
    "\u210d": "H",  # ℍ
    "\u210e": "h",  # ℎ
    "\u2110": "I",  # ℐ
    "\u2111": "I",  # ℑ
    "\u2112": "L",  # ℒ
    "\u2113": "l",  # ℓ
    "\u2115": "N",  # ℕ
    "\u2118": "P",  # ℘
    "\u2119": "P",  # ℙ
    "\u211a": "Q",  # ℚ
    "\u211b": "R",  # ℛ
    "\u211c": "R",  # ℜ
    "\u211d": "R",  # ℝ
    "\u2124": "Z",  # ℤ
    "\u2128": "Z",  # ℨ
    # Latin small capital letters
    "\u1d00": "A",  # ᴀ
    "\u1d04": "C",  # ᴄ
    "\u1d07": "E",  # ᴇ
    "\u0262": "G",  # ɢ
    "\u029c": "H",  # ʜ
    "\u026a": "I",  # ɪ
    "\u1d0a": "J",  # ᴊ
    "\u1d0b": "K",  # ᴋ
    "\u029f": "L",  # ʟ
    "\u1d0d": "M",  # ᴍ
    "\u0274": "N",  # ɴ
    "\u1d0f": "O",  # ᴏ
    "\u1d18": "P",  # ᴘ
    "\u0280": "R",  # ʀ
    "\u1d1b": "T",  # ᴛ
    "\u1d1c": "U",  # ᴜ
    "\u1d20": "V",  # ᴠ
    "\u1d21": "W",  # ᴡ
    "\u028f": "Y",  # ʏ
    "\u1d22": "Z",  # ᴢ
    # Other common lookalikes
    "\u0131": "i",  # ı (dotless i)
    "\u217c": "l",  # ⅼ (Roman numeral small l)
    "\u2170": "i",  # ⅰ
    "\u01b6": "z",  # ƶ
    "\u0455": "s",  # ѕ
}


def normalize_homoglyphs(text: str) -> str:
    """Replace Unicode lookalike characters with their ASCII equivalents."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def normalize_homoglyphs_series(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply normalize_homoglyphs element-wise.
    Returns (normalized_series, changed_bool_series).
    """
    normalized = series.apply(normalize_homoglyphs)
    changed = normalized != series
    return normalized, changed
