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

from spleen_bot.bot import Settings, SweeperBot
from spleen_bot.solver import solve
from spleen_bot.vision import FLAG, HIDDEN, find_continue, read_board


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


def self_test(root: Path) -> int:
    from spleen_bot.solver import solve as solve_grid

    # A 1 with a single hidden neighbor is a mine. A satisfied 1 opens the rest.
    forced = [[1, HIDDEN]]
    plan = solve_grid(forced)
    assert plan.flags == [(1, 0)], plan
    satisfied = [
        [FLAG, HIDDEN],
        [1, 1],
    ]
    plan = solve_grid(satisfied)
    assert (1, 0) in plan.safe_cells, plan

    temp = root / "docs" / "temp"
    expected = {
        "stage1.png": (9, 9),
        "stage2.png": (12, 11),
        "stage3.png": (15, 13),
        "stage4.png": (18, 14),
        "stage5.png": (20, 16),
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
        if name in {"stage3.png", "stage5.png"}:
            print(_render(board))
            moves = solve(board.cells)
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
    board = read_board(_load(path))
    if board is None:
        print("Papan tidak ketemu.")
        return 1
    print(f"Stage {board.stage} {board.cols}x{board.rows} origin=({board.origin_x},{board.origin_y})")
    print(_render(board))
    moves = solve(board.cells)
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
    parser.add_argument("--click-delay", type=float, default=0.008, help="Jeda antar klik, detik")
    parser.add_argument("--echo-mode", choices=("hover", "cast"), default="hover")
    parser.add_argument("--no-chord", action="store_true", help="Jangan klik angka; klik tiap petak aman")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.preview:
        return preview(args.preview)
    if args.play:
        settings = Settings(
            monitor=args.monitor,
            click_delay=args.click_delay,
            echo_mode=args.echo_mode,
        )
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
