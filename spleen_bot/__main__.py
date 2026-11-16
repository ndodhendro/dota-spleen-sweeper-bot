"""Run the bot, or check the reference screenshots without clicking.

Examples:
  python -m spleen_bot --self-test
  python -m spleen_bot --play
  python -m spleen_bot --play --echo-mode cast
  python -m spleen_bot --preview docs/temp/stage5.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from spleen_bot.bot import SweeperBot, load_settings
from spleen_bot.solver import FLAG as SOLVER_FLAG
from spleen_bot.solver import HIDDEN as SOLVER_HIDDEN
from spleen_bot.solver import solve
from spleen_bot.vision import FLAG, HIDDEN, find_continue, read_board, read_mine_counter


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _render(board) -> str:
    glyphs = {HIDDEN: ".", FLAG: "F", -3: "?"}
    lines = []
    for row in board.cells:
        lines.append("".join(glyphs.get(value, str(value)) for value in row))
    return "\n".join(lines)


def _consistent(board) -> list[str]:
    errors = []
    rows, cols = board.rows, board.cols
    for y in range(rows):
        for x in range(cols):
            number = board.cells[y][x]
            if number < 0:
                continue
            flags = hidden = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < cols and 0 <= ny < rows):
                        continue
                    value = board.cells[ny][nx]
                    if value == FLAG:
                        flags += 1
                    elif value == HIDDEN:
                        hidden += 1
            if flags > number or flags + hidden < number:
                errors.append(f"({x},{y})={number} flags={flags} hidden={hidden}")
    return errors


def _pressure(grid, extra_flags, extra_safe) -> tuple[int, int]:
    """How many numbers overflow or starve after these extra flags and opens."""
    rows = len(grid)
    cols = len(grid[0])
    mines = {(x, y) for y in range(rows) for x in range(cols) if grid[y][x] == FLAG}
    mines |= set(extra_flags)
    safe = set(extra_safe)
    over = starve = 0
    for y in range(rows):
        for x in range(cols):
            number = grid[y][x]
            if number < 0:
                continue
            flagged = unknown = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < cols and 0 <= ny < rows):
                        continue
                    if (nx, ny) in mines:
                        flagged += 1
                    elif grid[ny][nx] == HIDDEN and (nx, ny) not in mines and (nx, ny) not in safe:
                        unknown += 1
            if flagged > number:
                over += 1
            if flagged + unknown < number:
                starve += 1
    return over, starve


def self_test(root: Path) -> int:
    from spleen_bot.solver import solve as solve_grid

    # A 1 with a single hidden neighbor is a mine. A satisfied 1 opens the rest.
    forced = [[1, SOLVER_HIDDEN]]
    plan = solve_grid(forced)
    assert plan.flags == [(1, 0)], plan
    satisfied = [
        [SOLVER_FLAG, SOLVER_HIDDEN],
        [1, 1],
    ]
    plan = solve_grid(satisfied)
    assert (1, 0) in plan.safe_cells, plan

    # Mine counting: the 1 accounts for the only remaining mine, so the far
    # cells are safe even though no local pattern touches them.
    separated = [
        [1, SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN],
        [SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN],
    ]
    far = [(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1)]
    local = solve_grid(separated)
    assert not any(cell in local.safe_cells for cell in far), local
    counted = solve_grid(separated, 1)
    for cell in far:
        assert cell in counted.safe_cells, (cell, counted)
    assert solve_grid(separated, 0).safe_cells == []
    assert not solve_grid(separated, 50).has_progress
    # One mine is still outside the 1. Those far cells are 1/6, safer than the
    # frontier cells at 1/3, so the guess goes there.
    spread = solve_grid(separated, 2)
    assert spread.guess in far, spread
    assert 0.10 < spread.guess_risk < 0.25, spread

    # Combinations: putting a mine on (3, 0) forces a second mine, which does
    # not fit when only one mine remains, so that cell is safe and (2, 0) is
    # the mine. With two mines left the heavier placement is the one that fits.
    combo = [
        [SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN, SOLVER_HIDDEN],
        [SOLVER_FLAG, 2, 1, 1],
    ]
    alone = solve_grid(combo)
    assert (3, 0) not in alone.safe_cells, alone
    assert (2, 0) not in alone.flags, alone
    one_left = solve_grid(combo, 1)
    assert one_left.flags == [(2, 0)], one_left
    for cell in ((0, 0), (1, 0), (3, 0)):
        assert cell in one_left.safe_cells, (cell, one_left)
    two_left = solve_grid(combo, 2)
    assert sorted(two_left.flags) == [(0, 0), (3, 0)], two_left
    assert (2, 0) in two_left.safe_cells, two_left

    from spleen_bot.bot import load_settings

    loaded = load_settings(root / "config.ini")
    assert loaded.click_delay >= 0.05, loaded
    assert loaded.settle_timeout >= loaded.click_delay, loaded

    opening = [[SOLVER_HIDDEN] * 9 for _ in range(9)]
    first = solve_grid(opening, 10)
    assert not first.has_progress, first
    assert first.guess == (4, 4), first

    temp = root / "docs" / "temp"
    expected = {
        "stage1.png": (9, 9),
        "stage2.png": (12, 11),
        "stage3.png": (15, 13),
        "stage4.png": (18, 14),
        "stage5.png": (20, 16),
    }
    counters = {
        "stage1.png": 10,
        "stage2.png": 19,
        "stage3.png": 32,
        "stage4.png": 47,
        "stage5.png": 18,
    }
    failed = False
    for name, shape in expected.items():
        rgb = _load(temp / name)
        board = read_board(rgb)
        if board is None:
            print(f"FAIL {name}: papan tidak ketemu")
            failed = True
            continue
        got = (board.cols, board.rows)
        hidden = sum(value == HIDDEN for row in board.cells for value in row)
        flags = sum(value == FLAG for row in board.cells for value in row)
        numbers = sum(value > 0 for row in board.cells for value in row)
        zeros = sum(value == 0 for row in board.cells for value in row)
        print(
            f"{name}: {got[0]}x{got[1]} origin=({board.origin_x},{board.origin_y}) "
            f"pitch={board.pitch} H={hidden} F={flags} N={numbers} 0={zeros} items={len(board.pickups)}"
        )
        if got != shape:
            print(f"  FAIL ukuran, harus {shape[0]}x{shape[1]}")
            failed = True
        if name == "stage1.png" and (numbers or flags or zeros):
            print("  FAIL stage 1 harus masih tertutup semua")
            failed = True
        errors = _consistent(board)
        if errors:
            print(f"  FAIL logika lokal {len(errors)} contoh {errors[:4]}")
            failed = True
        mines_left = read_mine_counter(rgb, board)
        print(f"  sisa ranjau={mines_left}")
        if mines_left != counters[name]:
            print(f"  FAIL penghitung, harus {counters[name]}")
            failed = True
        moves = solve(board.cells, mines_left)
        before = _pressure(board.cells, [], [])
        after = _pressure(board.cells, moves.flags, moves.safe_cells)
        if after[0] > before[0] or after[1] > before[1]:
            print(f"  FAIL rencana merusak angka {before} -> {after}")
            failed = True
        if name == "stage1.png" and (moves.has_progress or moves.guess != (4, 4)):
            print(f"  FAIL stage 1 harus skill di tengah, dapat {moves}")
            failed = True
        if name in {"stage3.png", "stage5.png"}:
            print(_render(board))
            print(
                f"  rencana flag={len(moves.flags)} chord={len(moves.chords)} "
                f"klik={len(moves.clicks)} tebak={moves.guess} risiko={moves.guess_risk:.2f}"
            )
    won = _load(temp / "stagewon.png")
    button = find_continue(won)
    print(f"stagewon continue={button}")
    if button is None:
        print("  FAIL tombol CONTINUE tidak ketemu")
        failed = True
    plain = _load(temp / "stage1.png")
    if find_continue(plain) is not None:
        print("  FAIL stage 1 terbaca sebagai CONTINUE")
        failed = True
    if failed:
        print("SELF-TEST GAGAL")
        return 1
    print("SELF-TEST OK")
    return 0


def preview(path: Path) -> int:
    rgb = _load(path)
    board = read_board(rgb)
    if board is None:
        print("Papan tidak ketemu.")
        return 1
    mines_left = read_mine_counter(rgb, board)
    print(f"Stage {board.stage} {board.cols}x{board.rows} origin=({board.origin_x},{board.origin_y}) sisa={mines_left}")
    print(_render(board))
    moves = solve(board.cells, mines_left)
    print(f"flag {moves.flags}")
    print(f"chord {moves.chords}")
    print(f"klik {moves.clicks}")
    print(f"tebak {moves.guess} risiko {moves.guess_risk:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bot Spleen Sweeper Dota 2")
    parser.add_argument("--play", action="store_true", help="Mainkan papan yang sedang terbuka")
    parser.add_argument("--self-test", action="store_true", help="Cek screenshot referensi, tanpa klik")
    parser.add_argument("--preview", type=Path, help="Baca satu screenshot dan cetak rencananya")
    parser.add_argument("--monitor", type=int, default=1, help="Nomor monitor mss (1 = utama)")
    parser.add_argument("--config", type=Path, help="File jeda, bawaan config.ini di folder proyek")
    parser.add_argument("--click-delay", type=float, default=None, help="Timpa jeda antar klik dari config, detik")
    parser.add_argument("--echo-mode", choices=("hover", "cast"), default="hover")
    parser.add_argument("--no-chord", action="store_true", help="Jangan klik angka; klik tiap petak aman")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.preview:
        return preview(args.preview)
    if args.play:
        config_path = args.config or (root / "config.ini")
        try:
            settings = load_settings(
                config_path,
                monitor=args.monitor,
                echo_mode=args.echo_mode,
                click_delay=args.click_delay,
            )
        except ValueError as exc:
            print(exc)
            return 1
        bot = SweeperBot(settings)
        bot.chord_enabled = not args.no_chord
        try:
            bot.play()
        finally:
            bot.close()
        return 0
    return self_test(root)


if __name__ == "__main__":
    sys.exit(main())
