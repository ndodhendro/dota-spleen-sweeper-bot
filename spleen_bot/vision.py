"""Find the Spleen Sweeper board and read every tile."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spleen_bot.digits import ink_mask, read_digit

# Tile pitch on a 1920x1080 screenshot. Other scales are measured from the frame.
PITCH_HINT = 36

# Stage (columns, rows) in the order the minigame grows.
LAYOUTS: tuple[tuple[int, int], ...] = (
    (9, 9),
    (12, 11),
    (15, 13),
    (18, 14),
    (20, 16),
)

HIDDEN = -1
FLAG = -2
UNKNOWN = -3


@dataclass
class Board:
    cols: int
    rows: int
    origin_x: int
    origin_y: int
    pitch: int
    cells: list[list[int]]
    pickups: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def stage(self) -> int:
        try:
            return LAYOUTS.index((self.cols, self.rows)) + 1
        except ValueError:
            return 0

    def center(self, col: int, row: int) -> tuple[int, int]:
        half = self.pitch // 2
        return (
            self.origin_x + col * self.pitch + half,
            self.origin_y + row * self.pitch + half,
        )

    def signature(self) -> tuple:
        return tuple(tuple(row) for row in self.cells)


def _masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    grass = (r > 65) & (r < 130) & (g > 82) & (g < 130) & (b > 20) & (b < 55) & (g + 6 >= r) & ((g - b) > 38)
    dirt = (r > 62) & (r < 120) & (g > 45) & (g < 85) & (b > 26) & (b < 65) & (r > g + 6) & (g + 4 >= b)
    # Outer frame is a near-black brown line, distinct from the blue panel.
    dark = (r < 52) & (g < 44) & (b < 36) & ((r.astype(np.int32) + g + b) < 120)
    return grass, dirt, dark


def _longest_run(indexes: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    if len(indexes) == 0:
        return None
    best = (int(indexes[0]), int(indexes[0]), 1)
    start = prev = int(indexes[0])
    count = 1
    for value in indexes[1:]:
        value = int(value)
        if value - prev <= max_gap:
            count += value - prev
            prev = value
            if count > best[2]:
                best = (start, prev, count)
        else:
            start = prev = value
            count = 1
    return best[0], best[1]


def _content_bbox(grass: np.ndarray, dirt: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = grass.shape
    # Ignore the train UI outside the cauldron window.
    region = np.zeros_like(grass)
    region[int(h * 0.18) : int(h * 0.88), int(w * 0.22) : int(w * 0.78)] = True
    grass = grass & region
    col = grass.sum(axis=0)
    row = grass.sum(axis=1)
    if col.max() < 80 or row.max() < 80:
        return None
    xs = np.flatnonzero(col > max(40, int(col.max() * 0.18)))
    ys = np.flatnonzero(row > max(40, int(row.max() * 0.18)))
    span_x = _longest_run(xs, max_gap=10)
    span_y = _longest_run(ys, max_gap=10)
    if span_x is None or span_y is None:
        return None
    x0, x1 = span_x
    y0, y1 = span_y
    # Grow through revealed dirt until the dark frame. Dirt outside the frame
    # is separated by that frame, so the walk stops.
    content = grass | dirt

    def column_ratio(x: int) -> float:
        if x < 0 or x >= w:
            return 0.0
        return float(content[y0 : y1 + 1, x].mean())

    def row_ratio(y: int) -> float:
        if y < 0 or y >= h:
            return 0.0
        return float(content[y, x0 : x1 + 1].mean())

    # A revealed side of the board can be hundreds of pixels of dirt, so the
    # walk has to be allowed to cross the whole panel. The dark frame stops it.
    limit = max(w, h)
    guard = 0
    while x0 > 0 and column_ratio(x0 - 1) > 0.35 and guard < limit:
        x0 -= 1
        guard += 1
    guard = 0
    while x1 + 1 < w and column_ratio(x1 + 1) > 0.35 and guard < limit:
        x1 += 1
        guard += 1
    guard = 0
    while y0 > 0 and row_ratio(y0 - 1) > 0.35 and guard < limit:
        y0 -= 1
        guard += 1
    guard = 0
    while y1 + 1 < h and row_ratio(y1 + 1) > 0.35 and guard < limit:
        y1 += 1
        guard += 1
    return x0, y0, x1 + 1, y1 + 1


def _match_layout(width: int, height: int) -> tuple[int, int, int] | None:
    best: tuple[float, int, int, int] | None = None
    for cols, rows in LAYOUTS:
        pitch_x = width / cols
        pitch_y = height / rows
        if not (30 <= pitch_x <= 48 and 30 <= pitch_y <= 48):
            continue
        if abs(pitch_x - pitch_y) > 2.0:
            continue
        pitch = int(round((pitch_x + pitch_y) / 2))
        error = abs(width - cols * pitch) + abs(height - rows * pitch)
        if best is None or error < best[0]:
            best = (error, cols, rows, pitch)
    if best is None or best[0] > 16:
        return None
    return best[1], best[2], best[3]


def _pink(cell: np.ndarray) -> int:
    r = cell[:, :, 0].astype(np.int16)
    g = cell[:, :, 1].astype(np.int16)
    b = cell[:, :, 2].astype(np.int16)
    # Mine flags are pink. Flowers on hidden tiles are orange and fail b > 75.
    return int(((r > 165) & (b > 75) & (g < 155) & (b + 15 > g) & (r > g + 40)).sum())


def _grass_ratio(cell: np.ndarray) -> float:
    r = cell[:, :, 0].astype(np.int16)
    g = cell[:, :, 1].astype(np.int16)
    b = cell[:, :, 2].astype(np.int16)
    grass = (g > 80) & (b < 58) & ((g - b) > 32) & (np.abs(g - r) < 42) & (r > 55)
    return float(grass.mean())


def _dirt_ratio(cell: np.ndarray) -> float:
    r = cell[:, :, 0].astype(np.int16)
    g = cell[:, :, 1].astype(np.int16)
    b = cell[:, :, 2].astype(np.int16)
    dirt = (r > 60) & (r < 125) & (g > 42) & (g < 90) & (b < 70) & (r > g) & (g + 6 >= b)
    return float(dirt.mean())


def _pickup(cell: np.ndarray) -> str | None:
    """Mana vials and clocks spawn on top of a tile and despawn quickly."""
    inner = cell[6:-6, 6:-6]
    if inner.size == 0:
        return None
    r = inner[:, :, 0].astype(np.int16)
    g = inner[:, :, 1].astype(np.int16)
    b = inner[:, :, 2].astype(np.int16)
    # Cyan vials, not the pure-blue cursor (that cursor has a low green channel).
    cyan = (b > 185) & (g > 175) & (r < 120) & (b > r + 50) & (abs(g - b) < 50)
    gold = (r > 200) & (g > 175) & (b < 80) & (r + g > 400)
    # Numbers are lime, not gold, and they sit on dirt. Require a real blob.
    if int(cyan.sum()) > 28 and _grass_ratio(cell) > 0.25:
        return "mana"
    if int(gold.sum()) > 28 and _grass_ratio(cell) > 0.25 and int(ink_mask(cell).sum()) < 10:
        return "clock"
    return None


def _read_cell(cell: np.ndarray) -> int:
    if _pink(cell) > 14:
        return FLAG
    inner = cell[4:-4, 4:-4] if cell.shape[0] > 16 else cell
    digit = read_digit(inner)
    if digit is not None:
        return digit
    # Real ink on dirt that is not 1-4 yet (a 5, a cursor, a partial frame).
    # Never call that an empty zero: a false zero makes the solver click mines.
    if int(ink_mask(inner).sum()) >= 36 and _dirt_ratio(cell) > 0.25:
        return UNKNOWN
    if _dirt_ratio(cell) > 0.45 and _grass_ratio(cell) < 0.25:
        return 0
    return HIDDEN


def read_board(rgb: np.ndarray) -> Board | None:
    """Parse a full screenshot. rgb is HxWx3 uint8."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    rgb = rgb[:, :, :3]
    grass, dirt, _dark = _masks(rgb)
    box = _content_bbox(grass, dirt)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    matched = _match_layout(x1 - x0, y1 - y0)
    if matched is None:
        return None
    cols, rows, pitch = matched
    # The top-left content pixel is the sharp edge. Snap size to the lattice
    # so a short bottom edge (darker last row) does not shift every tile.
    origin_x, origin_y = x0, y0
    cells: list[list[int]] = []
    pickups: list[tuple[int, int, str]] = []
    height, width = rgb.shape[:2]
    for row in range(rows):
        line: list[int] = []
        for col in range(cols):
            left = origin_x + col * pitch
            top = origin_y + row * pitch
            right = min(width, left + pitch)
            bottom = min(height, top + pitch)
            if left < 0 or top < 0 or right - left < pitch // 2 or bottom - top < pitch // 2:
                return None
            cell = rgb[top:bottom, left:right]
            kind = _pickup(cell)
            if kind is not None:
                pickups.append((col, row, kind))
            line.append(_read_cell(cell))
        cells.append(line)
    return Board(cols, rows, origin_x, origin_y, pitch, cells, pickups)


def _find_overlay_button(rgb: np.ndarray) -> tuple[int, int] | None:
    """Center of the blue Stage Won / You Lost button, if that overlay is up."""
    height, width = rgb.shape[:2]
    y0, y1 = int(height * 0.46), int(height * 0.68)
    x0, x1 = int(width * 0.36), int(width * 0.64)
    sub = rgb[y0:y1, x0:x1]
    r = sub[:, :, 0].astype(np.int16)
    g = sub[:, :, 1].astype(np.int16)
    b = sub[:, :, 2].astype(np.int16)
    blue = (r < 70) & (g > 40) & (g < 150) & (b > 100) & (b < 210) & (b > g + 25) & (b > r + 55)
    if int(blue.sum()) < 600:
        return None
    ys, xs = np.nonzero(blue)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    if (right - left) < 90 or not (24 <= (bottom - top) <= 90):
        return None
    above = rgb[max(0, y0 + top - 48) : y0 + top, x0 + left : x0 + right + 1]
    if above.size == 0:
        return None
    white = (
        (above[:, :, 0] > 210)
        & (above[:, :, 1] > 210)
        & (above[:, :, 2] > 210)
    )
    if int(white.sum()) < 25:
        return None
    return x0 + (left + right) // 2, y0 + (top + bottom) // 2


def find_continue(rgb: np.ndarray) -> tuple[int, int] | None:
    """Center of CONTINUE after a stage clear. PLAY AGAIN sits lower and is ignored."""
    button = _find_overlay_button(rgb)
    if button is None or button[1] > rgb.shape[0] * 0.59:
        return None
    return button


def find_play_again(rgb: np.ndarray) -> tuple[int, int] | None:
    """Center of PLAY AGAIN on the You Lost screen. That restarts at stage 1."""
    button = _find_overlay_button(rgb)
    if button is None or button[1] <= rgb.shape[0] * 0.59:
        return None
    return button


def find_play(rgb: np.ndarray) -> tuple[int, int] | None:
    """Center of the menu PLAY button. LEAVE is the button to its right."""
    height, width = rgb.shape[:2]
    y0, y1 = int(height * 0.74), int(height * 0.84)
    x0, x1 = int(width * 0.38), int(width * 0.62)
    sub = rgb[y0:y1, x0:x1]
    red = sub[:, :, 0].astype(np.int16)
    green = sub[:, :, 1].astype(np.int16)
    blue = sub[:, :, 2].astype(np.int16)
    fill = (red < 50) & (green > 60) & (green < 140) & (blue > 140) & (blue < 200) & (blue > red + 80) & (blue > green + 40)
    columns = fill.sum(axis=0) > 8
    runs: list[tuple[int, int]] = []
    start = None
    for index, on in enumerate(columns):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start > 50:
                runs.append((start, index))
            start = None
    if start is not None and len(columns) - start > 50:
        runs.append((start, len(columns)))
    buttons: list[tuple[int, int]] = []
    for run_start, run_end in runs:
        glyph = fill[:, run_start:run_end]
        ys = np.nonzero(glyph.any(axis=1))[0]
        if len(ys) == 0 or not (24 <= int(ys.max() - ys.min()) <= 60):
            continue
        patch = sub[int(ys.min()) : int(ys.max()) + 1, run_start:run_end]
        white = (patch[:, :, 0] > 200) & (patch[:, :, 1] > 200) & (patch[:, :, 2] > 190)
        if int(white.sum()) < 40:
            continue
        buttons.append((x0 + (run_start + run_end) // 2, y0 + (int(ys.min()) + int(ys.max())) // 2))
    if len(buttons) < 2:
        return None
    buttons.sort()
    play, leave = buttons[0], buttons[1]
    if leave[0] - play[0] > 250:
        return None
    return play


# HUD digits above the board, normalized to 15x10 from the reference shots.
# 5 and 6 are drawn in the same stroke; a weak match is rejected rather than guessed.
_COUNTER_ROWS: dict[int, list[str]] = {
    0: [
        "...#####..",
        "..#....##.",
        ".##.....#.",
        ".#......##",
        ".#......##",
        "##.......#",
        "##.......#",
        "##.......#",
        "##.......#",
        "##......##",
        ".#......##",
        ".#......#.",
        ".##.....#.",
        "..##...#..",
        "...####...",
    ],
    1: [
        "......####",
        "##########",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
        ".........#",
    ],
    2: [
        "#######...",
        "##....##..",
        ".......##.",
        ".......##.",
        "........#.",
        ".......##.",
        ".......#..",
        "......##..",
        "......#...",
        ".....#....",
        "....##....",
        "...##.....",
        "..##......",
        "###.......",
        "##########",
    ],
    3: [
        "########..",
        ".......##.",
        "........##",
        "........##",
        "........#.",
        "........#.",
        "......#...",
        "...####...",
        ".......##.",
        "........##",
        ".........#",
        "........##",
        "........##",
        ".......##.",
        "########..",
    ],
    4: [
        ".......#..",
        "......##..",
        ".....###..",
        ".....#.#..",
        "....#..#..",
        "...#...#..",
        "..#....#..",
        "..#....#..",
        ".#.....#..",
        "##########",
        "......###.",
        ".......#..",
        ".......#..",
        ".......#..",
        ".......#..",
    ],
    5: [
        "##########",
        "##........",
        "##........",
        "##........",
        "########..",
        ".......##.",
        "........##",
        "........#.",
        "........#.",
        "........##",
        ".......##.",
        "......##..",
        ".....##...",
        "...###....",
        "..##......",
    ],
    6: [
        "..##......",
        "....#.....",
        ".....#....",
        "......#...",
        "......##..",
        ".......##.",
        "...######.",
        ".###...##.",
        ".#......##",
        "##......##",
        "##.......#",
        "##.......#",
        "##......##",
        ".##....##.",
        "..##..##..",
    ],
    7: [
        "##########",
        "........##",
        "........#.",
        ".......##.",
        ".......#..",
        "......##..",
        "......#...",
        ".....##...",
        ".....#....",
        ".....#....",
        "....##....",
        "....#.....",
        "...##.....",
        "...#......",
        "..##......",
    ],
    8: [
        "...#####..",
        ".##....##.",
        ".#......#.",
        ".#......#.",
        ".##.....#.",
        "..#....##.",
        "...####...",
        "...####...",
        "..#....##.",
        ".#......##",
        "##......##",
        "##......##",
        ".#......##",
        ".##....##.",
        "...#####..",
    ],
    9: [
        "..##..##..",
        ".##....##.",
        "##......##",
        "##.......#",
        "##.......#",
        "##......##",
        ".#......##",
        ".###...##.",
        "...######.",
        ".......##.",
        "......##..",
        "......#...",
        ".....#....",
        "....#.....",
        "..##......",
    ],
}
_COUNTER_TEMPLATES: dict[int, np.ndarray] = {}


def _counter_templates() -> dict[int, np.ndarray]:
    if _COUNTER_TEMPLATES:
        return _COUNTER_TEMPLATES
    for digit, rows in _COUNTER_ROWS.items():
        _COUNTER_TEMPLATES[digit] = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    return _COUNTER_TEMPLATES


def _glyph_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _normalize_counter(glyph: np.ndarray) -> np.ndarray:
    height, width = glyph.shape
    src_y = np.linspace(0, height - 1, 15).astype(int)
    src_x = np.linspace(0, width - 1, 10).astype(int)
    return glyph[src_y][:, src_x]


def _classify_counter_digit(glyph: np.ndarray) -> int | None:
    sample = _normalize_counter(glyph)
    scored = [(_glyph_jaccard(sample, template), digit) for digit, template in _counter_templates().items()]
    scored.sort(reverse=True)
    best, second = scored[0], scored[1]
    if best[0] < 0.70 or best[0] - second[0] < 0.12:
        return None
    return best[1]


def read_mine_counter(rgb: np.ndarray, board: Board) -> int | None:
    """Remaining mines shown at the right of the HUD above the board.

    The value falls as the game accepts a flag. None when the digits are
    missing or not confident, and the solver then stays on local logic.
    """
    height, width = rgb.shape[:2]
    scale = board.pitch / float(PITCH_HINT)
    top = int(round(board.origin_y - 80 * scale))
    bottom = int(round(board.origin_y - 36 * scale))
    right = board.origin_x + board.cols * board.pitch
    left = int(round(right - 56 * scale))
    top = max(0, top)
    left = max(0, left)
    bottom = min(height, bottom)
    right_edge = min(width, right + 6)
    if bottom <= top or right_edge <= left:
        return None
    crop = rgb[top:bottom, left:right_edge]
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    white = (red > 210) & (green > 210) & (blue > 200) & (np.abs(red - blue) < 40)
    columns = white.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for index, on in enumerate(columns):
        if on and start is None:
            start = index
        elif not on and start is not None:
            if index - start >= 3:
                runs.append((start, index))
            start = None
    if start is not None and len(columns) - start >= 3:
        runs.append((start, len(columns)))
    if not runs:
        return None

    gap_limit = max(6, int(round(8 * scale)))
    cluster = [runs[-1]]
    for run in reversed(runs[:-1]):
        if cluster[0][0] - run[1] > gap_limit:
            break
        cluster.insert(0, run)

    digits: list[int] = []
    min_height = max(8, int(round(10 * scale)))
    max_height = int(round(22 * scale))
    for run_start, run_end in cluster:
        glyph = white[:, run_start:run_end]
        ys, xs = np.nonzero(glyph)
        if len(xs) == 0:
            return None
        glyph = glyph[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        glyph_h, glyph_w = glyph.shape
        if not (min_height <= glyph_h <= max_height and 3 <= glyph_w <= int(round(16 * scale))):
            return None
        digit = _classify_counter_digit(glyph)
        if digit is None:
            return None
        digits.append(digit)
    if not digits or len(digits) > 3:
        return None
    value = int("".join(str(digit) for digit in digits))
    if value > board.cols * board.rows:
        return None
    return value


def read_mana(rgb: np.ndarray, board: Board) -> int:
    """Approximate mana 0-250 from the blue bar under the board.

    On the reference 1920x1080 UI the fill is about one pixel per mana,
    starting just right of the Echo Slash icon.
    """
    height, width = rgb.shape[:2]
    bar_top = board.origin_y + board.rows * board.pitch + 8
    bar_bottom = min(height - 1, bar_top + 70)
    if bar_bottom <= bar_top:
        return 0
    # The bar stays centered on the screen as the grid grows.
    center = width // 2
    # Icon + bar widget measured on the reference shots: fill begins ~100px
    # left of center? Full bar on stage 1 ran from x=860 to x=1112 at width 1919,
    # i.e. center-100 to center+152. Scale with width.
    scale = width / 1919.0
    fill_left = int(center - 100 * scale)
    fill_right = int(center + 153 * scale)
    strip = rgb[bar_top:bar_bottom, max(0, fill_left) : min(width, fill_right)]
    if strip.size == 0:
        return 0
    r = strip[:, :, 0].astype(np.int16)
    g = strip[:, :, 1].astype(np.int16)
    b = strip[:, :, 2].astype(np.int16)
    lit = (b > 140) & (b > r + 50) & (g > 40) & (g < 200)
    cols = np.flatnonzero(lit.any(axis=0))
    if len(cols) == 0:
        return 0
    # Ignore a detached icon blob: keep the rightmost contiguous run.
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for value in cols[1:]:
        value = int(value)
        if value - prev <= 3:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    start, end = max(runs, key=lambda item: item[1] - item[0])
    full = max(1, fill_right - fill_left)
    pixels = end - start + 1
    return int(max(0, min(250, round(250 * pixels / (full * 0.82)))))
