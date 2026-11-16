"""Play loop: read the board, click the forced tiles, Echo Slash when stuck."""
from __future__ import annotations

import configparser
import time
from dataclasses import dataclass
from pathlib import Path

import mss
import numpy as np

from spleen_bot import input as win
from spleen_bot.solver import Plan, solve
from spleen_bot.vision import FLAG, HIDDEN, Board, find_continue, read_board, read_mana, read_mine_counter

# Fresh-stage mine totals from the reference screenshots. Only used as a hint
# in the log; the solver does not trust them for clicks because a run can differ.
STAGE_MINES = {1: 10, 2: 19, 4: 47}


@dataclass
class Settings:
    monitor: int = 1
    click_delay: float = 0.07
    flag_pause: float = 0.12
    settle_timeout: float = 0.60
    echo_settle: float = 0.85
    echo_aim: float = 0.08
    echo_cast_pause: float = 0.12
    park_pause: float = 0.04
    continue_pause: float = 0.50
    start_delay: float = 3.0
    echo_mode: str = "hover"  # hover = key 1 on the tile; cast = key 1 then left click
    mana_reserve: int = 18
    config_path: Path | None = None


# Nama di config.ini -> field Settings. Nilainya detik.
_DELAY_FIELDS = {
    "klik": "click_delay",
    "setelah_bendera": "flag_pause",
    "tunggu_papan": "settle_timeout",
    "tunggu_skill": "echo_settle",
    "bidik_skill": "echo_aim",
    "skill_lalu_klik": "echo_cast_pause",
    "singkirkan_kursor": "park_pause",
    "setelah_continue": "continue_pause",
    "mulai": "start_delay",
}


def _seconds(text: str) -> float:
    return float(text.strip().replace(",", "."))


def load_settings(
    path: Path,
    *,
    monitor: int = 1,
    echo_mode: str = "hover",
    click_delay: float | None = None,
) -> Settings:
    """Read action delays from config.ini. Missing keys keep the built-in values."""
    settings = Settings(monitor=monitor, echo_mode=echo_mode, config_path=path)
    if path.is_file():
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        section = parser["jeda"] if parser.has_section("jeda") else {}
        for key, field in _DELAY_FIELDS.items():
            if key not in section:
                continue
            try:
                value = _seconds(section[key])
            except ValueError:
                raise ValueError(f"{path.name}: jeda '{key}' harus angka detik, dapat {section[key]!r}.") from None
            if value < 0:
                raise ValueError(f"{path.name}: jeda '{key}' tidak boleh negatif.")
            setattr(settings, field, value)
    if click_delay is not None:
        settings.click_delay = click_delay
    return settings


class SweeperBot:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.sct = mss.mss()
        self.mon = self.sct.monitors[self.settings.monitor]
        self.chord_enabled = True
        self.stopped = False
        self.echoes = 0
        self.clicks = 0

    def close(self) -> None:
        self.sct.close()

    def grab(self) -> np.ndarray:
        shot = self.sct.grab(self.mon)
        # mss is BGRA. Drop alpha and swap to RGB.
        frame = np.asarray(shot, dtype=np.uint8)
        return frame[:, :, :3][:, :, ::-1].copy()

    def screen_point(self, x: int, y: int) -> tuple[int, int]:
        return x + int(self.mon["left"]), y + int(self.mon["top"])

    def park(self, board: Board) -> None:
        # Sit on the dark frame so the cursor does not cover a tile.
        x = board.origin_x - 14
        y = board.origin_y + board.pitch // 2
        if x < 4:
            x = board.origin_x + board.cols * board.pitch + 14
        win.move(*self.screen_point(x, y))

    def _abort(self) -> bool:
        if win.abort_pressed():
            self.stopped = True
            print("Berhenti (Esc/F8).")
            return True
        return False

    def capture_board(self) -> tuple[np.ndarray, Board | None]:
        frame = self.grab()
        return frame, read_board(frame)

    def wait_stable(self, previous: tuple | None, timeout: float) -> tuple[np.ndarray, Board | None]:
        """Return once two reads agree, or the board differs from `previous`."""
        deadline = time.perf_counter() + timeout
        last_sig = None
        last = self.capture_board()
        while time.perf_counter() < deadline:
            if self._abort():
                return last
            frame, board = self.capture_board()
            sig = None if board is None else board.signature()
            if board is not None and sig == last_sig and (previous is None or sig != previous):
                return frame, board
            last_sig = sig
            last = (frame, board)
            time.sleep(0.02)
        return last

    def echo(self, board: Board, col: int, row: int) -> None:
        x, y = self.screen_point(*board.center(col, row))
        win.move(x, y)
        time.sleep(self.settings.echo_aim)
        win.tap(win.VK_1)
        if self.settings.echo_mode == "cast":
            time.sleep(self.settings.echo_cast_pause)
            win.click("left")
        self.echoes += 1

    def act(self, board: Board, plan: Plan, frame: np.ndarray, mines_left: int | None = None) -> str:
        pickup_clicks: list[tuple[int, int]] = []
        pickup_flags: list[tuple[int, int]] = []
        echo_pickup: tuple[int, int] | None = None
        safe = {(c, r) for c, r in plan.clicks}
        safe.update(_opened_by(board, plan))
        mine_cells = {(c, r) for c, r in plan.flags}
        for col, row, kind in board.pickups:
            value = board.cells[row][col]
            if value == HIDDEN and (col, row) in safe:
                pickup_clicks.append(board.center(col, row))
            elif value == FLAG or (col, row) in mine_cells:
                pickup_flags.append(board.center(col, row))
            elif kind == "clock" and value == HIDDEN:
                echo_pickup = (col, row)

        points = [self.screen_point(*point) for point in pickup_clicks]
        if points:
            print(f"  ambil {len(points)} item di petak aman")
            if not win.burst(points, "left", self.settings.click_delay):
                self.stopped = True
                return "abort"
            self.clicks += len(points)

        # Flags are only worth the click when a number-chord needs them.
        # A pickup sitting on a mine is nudged with one right click.
        if self.chord_enabled and (plan.flags or plan.chords):
            if plan.flags or pickup_flags:
                seen: set[tuple[int, int]] = set()
                flag_points: list[tuple[int, int]] = []
                for col, row in plan.flags:
                    point = self.screen_point(*board.center(col, row))
                    if point not in seen:
                        seen.add(point)
                        flag_points.append(point)
                for point in pickup_flags:
                    screen = self.screen_point(*point)
                    if screen not in seen:
                        seen.add(screen)
                        flag_points.append(screen)
                if flag_points and not win.burst(flag_points, "right", self.settings.click_delay):
                    self.stopped = True
                    return "abort"
                self.clicks += len(flag_points)
                time.sleep(self.settings.flag_pause)
            if plan.chords:
                chord_points = [self.screen_point(*board.center(c, r)) for c, r in plan.chords]
                if not win.burst(chord_points, "left", self.settings.click_delay):
                    self.stopped = True
                    return "abort"
                self.clicks += len(chord_points)
            if plan.clicks:
                points = [self.screen_point(*board.center(c, r)) for c, r in plan.clicks]
                if not win.burst(points, "left", self.settings.click_delay):
                    self.stopped = True
                    return "abort"
                self.clicks += len(points)
            return "chord" if plan.chords else "click"

        direct = plan.safe_cells or plan.clicks
        if direct:
            points = [self.screen_point(*board.center(c, r)) for c, r in direct]
            if not win.burst(points, "left", self.settings.click_delay):
                self.stopped = True
                return "abort"
            self.clicks += len(points)
            return "click"

        if _untouched(board) and plan.guess is not None:
            col, row = plan.guess
            print(f"  klik kiri pembuka di ({col},{row})")
            if not win.burst([self.screen_point(*board.center(col, row))], "left", 0):
                self.stopped = True
                return "abort"
            self.clicks += 1
            return "click"

        target = echo_pickup or plan.guess
        if target is None:
            return "done"
        mana = read_mana(frame, board)
        col, row = target
        if mana >= self.settings.mana_reserve:
            why = "jam" if echo_pickup else "buntu"
            left = f" sisa~{mines_left}" if mines_left is not None else ""
            print(f"  skill 1 di ({col},{row}) risiko {plan.guess_risk:.0%} mana~{mana}{left} ({why})")
            self.echo(board, col, row)
            return "echo"
        print(f"  mana habis (~{mana}), tebak biasa di ({col},{row}) risiko {plan.guess_risk:.0%}")
        if not win.burst([self.screen_point(*board.center(col, row))], "left", 0):
            self.stopped = True
            return "abort"
        self.clicks += 1
        return "guess"

    def play(self) -> None:
        pace = self.settings
        print("Spleen Sweeper bot. Fokuskan jendela Dota (borderless 1920x1080).")
        source = pace.config_path.name if pace.config_path and pace.config_path.is_file() else "bawaan"
        print(
            f"Jeda dari {source}: klik {pace.click_delay:.3f}s, setelah bendera {pace.flag_pause:.3f}s, "
            f"tunggu papan {pace.settle_timeout:.2f}s, tunggu skill {pace.echo_settle:.2f}s."
        )
        print(f"Mulai dalam {pace.start_delay:g} detik. Esc atau F8 untuk berhenti.")
        time.sleep(pace.start_delay)
        misses = 0
        last_stage = 0
        stalled: tuple | None = None
        stall_count = 0
        while not self.stopped:
            if self._abort():
                break
            frame = self.grab()
            button = find_continue(frame)
            if button is not None:
                print("Stage selesai, klik CONTINUE.")
                if not win.burst([self.screen_point(*button)], "left", 0):
                    break
                time.sleep(self.settings.continue_pause)
                stalled = None
                continue

            board = read_board(frame)
            if board is None:
                misses += 1
                if misses == 1:
                    print("Papan tidak kelihatan, menunggu...")
                if misses > 80:
                    print("Papan hilang terlalu lama. Berhenti.")
                    break
                time.sleep(0.05)
                continue
            misses = 0
            if board.stage != last_stage:
                mines = STAGE_MINES.get(board.stage)
                extra = f", ranjau awal {mines}" if mines else ""
                print(f"Stage {board.stage}: {board.cols}x{board.rows}{extra}")
                last_stage = board.stage
                stall_count = 0

            self.park(board)
            time.sleep(self.settings.park_pause)
            frame, board = self.capture_board()
            if board is None:
                continue

            mines_left = read_mine_counter(frame, board)
            plan = solve(board.cells, mines_left)
            signature = board.signature()
            action = self.act(board, plan, frame, mines_left)
            self.park(board)
            if action == "abort":
                break
            if action == "done":
                print("Tidak ada petak tersisa. Menunggu stage won...")
                time.sleep(0.2)
                continue

            timeout = self.settings.echo_settle if action == "echo" else self.settings.settle_timeout
            _frame, updated = self.wait_stable(signature, timeout)
            new_sig = None if updated is None else updated.signature()
            if new_sig == signature:
                stall_count += 1
                if action == "echo" and self.settings.echo_mode == "hover":
                    print("  skill 1 belum membuka petak. Berikutnya: tombol 1 lalu klik kiri.")
                    self.settings.echo_mode = "cast"
                    stall_count = 0
                elif action == "chord" and stall_count >= 1:
                    print("  chord tidak mengubah papan, beralih ke klik kiri langsung.")
                    self.chord_enabled = False
                if stall_count >= 4:
                    print("Papan tidak berubah setelah beberapa aksi. Berhenti agar tidak klik ganda.")
                    break
            else:
                stall_count = 0
            stalled = new_sig

        print(f"Selesai. Klik {self.clicks}, Echo Slash {self.echoes}.")


def _untouched(board: Board) -> bool:
    """A stage that just started has no revealed tiles, so the first click is blind."""
    return all(value == HIDDEN for row in board.cells for value in row)


def _opened_by(board: Board, plan: Plan) -> set[tuple[int, int]]:
    """Hidden tiles a chord would reveal. They are already known safe."""
    opened: set[tuple[int, int]] = set()
    goals = set(plan.chords)
    rows = board.rows
    cols = board.cols
    for y in range(rows):
        for x in range(cols):
            if (x, y) not in goals:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if dx == dy == 0:
                        continue
                    if 0 <= nx < cols and 0 <= ny < rows and board.cells[ny][nx] == HIDDEN:
                        opened.add((nx, ny))
    return opened
