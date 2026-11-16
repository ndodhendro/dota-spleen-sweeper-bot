"""Read the lime number on a revealed tile.

Templates are the 16x12 normalized glyphs from the reference screenshots.
A tile with real ink that matches none of them stays unknown so the solver
will not treat it as an empty zero.
"""
from __future__ import annotations

import numpy as np

INK_G = 145
INK_R = 90

# Width is forced to 12, so a thin "1" becomes a solid stem. Detect 1 before
# normalizing, from the unscaled bounding box.
_TEMPLATES: dict[int, np.ndarray] = {}


def _bits(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _load_templates() -> dict[int, np.ndarray]:
    if _TEMPLATES:
        return _TEMPLATES
    raw = {
        2: [
            "....###.....",
            "#########...",
            "##########..",
            "###########.",
            ".......####.",
            ".......####.",
            ".......####.",
            ".......####.",
            ".......###..",
            ".....#####..",
            "....#####...",
            "...#####....",
            "..#####.....",
            "############",
            "############",
            "############",
        ],
        3: [
            ".....#......",
            "##########..",
            "###########.",
            ".......#####",
            ".......#####",
            ".......#####",
            ".......####.",
            "...#######..",
            "...#######..",
            "......#####.",
            ".........###",
            ".........###",
            ".......#####",
            "############",
            "###########.",
            "##########..",
        ],
        4: [
            ".........#..",
            ".........#..",
            "........##..",
            ".......###..",
            ".....#####..",
            "....#...##..",
            "........##..",
            "........##..",
            "........##..",
            "..#.....##..",
            "############",
            "############",
            "........##..",
            "........##..",
            "........##..",
            "........##..",
        ],
    }
    for digit, rows in raw.items():
        _TEMPLATES[digit] = _bits(rows)
    return _TEMPLATES


def ink_mask(cell: np.ndarray) -> np.ndarray:
    r = cell[:, :, 0].astype(np.int16)
    g = cell[:, :, 1].astype(np.int16)
    b = cell[:, :, 2].astype(np.int16)
    return (g > INK_G) & (r > INK_R) & (b < 130) & (g > b + 40)


def _normalize(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    glyph = mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
    height, width = glyph.shape
    if height < 8:
        return None
    src_y = np.linspace(0, height - 1, 16).astype(int)
    src_x = np.linspace(0, width - 1, 12).astype(int)
    return glyph[src_y][:, src_x]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def read_digit(cell: np.ndarray) -> int | None:
    """Return 1-4, or None when the glyph is missing or not one of those shapes."""
    mask = ink_mask(cell)
    ink = int(mask.sum())
    # Flowers and grass sparkles stay under ~30. A cursor blob is much larger.
    if ink < 36 or ink > 160:
        return None
    ys, xs = np.nonzero(mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    if width <= 8 and height >= 12:
        return 1
    norm = _normalize(mask)
    if norm is None:
        return None
    best_digit = None
    best_score = 0.0
    for digit, template in _load_templates().items():
        score = _jaccard(norm, template)
        if score > best_score:
            best_score = score
            best_digit = digit
    if best_score >= 0.62:
        return best_digit
    return None
