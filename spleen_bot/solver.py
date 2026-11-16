"""Minesweeper deductions aimed at the fewest clicks.

Flags and chord clicks are preferred: one click on a satisfied number opens
every safe neighbor, which is faster than clicking those tiles one by one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HIDDEN = -1
FLAG = -2
UNKNOWN = -3


@dataclass
class Plan:
    flags: list[tuple[int, int]] = field(default_factory=list)
    chords: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[tuple[int, int]] = field(default_factory=list)
    safe_cells: list[tuple[int, int]] = field(default_factory=list)
    guess: tuple[int, int] | None = None
    guess_risk: float = 1.0

    @property
    def has_progress(self) -> bool:
        return bool(self.flags or self.chords or self.clicks)


def _neighbors(cols: int, rows: int, col: int, row: int) -> list[tuple[int, int]]:
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x, y = col + dx, row + dy
            if 0 <= x < cols and 0 <= y < rows:
                out.append((x, y))
    return out


def solve(grid: list[list[int]]) -> Plan:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    mines: set[tuple[int, int]] = set()
    safe: set[tuple[int, int]] = set()
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == FLAG:
                mines.add((x, y))

    def hidden_unknown(x: int, y: int) -> bool:
        return grid[y][x] == HIDDEN and (x, y) not in mines and (x, y) not in safe

    def overflows(extra: set[tuple[int, int]]) -> bool:
        trial = mines | extra
        for yy in range(rows):
            for xx in range(cols):
                number = grid[yy][xx]
                if number < 0:
                    continue
                flagged = sum(1 for cell in _neighbors(cols, rows, xx, yy) if cell in trial)
                if flagged > number:
                    return True
        return False

    def starves(extra_safe: set[tuple[int, int]]) -> bool:
        trial_safe = safe | extra_safe
        for yy in range(rows):
            for xx in range(cols):
                number = grid[yy][xx]
                if number < 0:
                    continue
                flagged = 0
                unknown = 0
                for cell in _neighbors(cols, rows, xx, yy):
                    if cell in mines:
                        flagged += 1
                    elif grid[cell[1]][cell[0]] == HIDDEN and cell not in mines and cell not in trial_safe:
                        unknown += 1
                if flagged + unknown < number:
                    return True
        return False

    changed = True
    while changed:
        changed = False
        constraints: list[tuple[frozenset[tuple[int, int]], int]] = []
        for y in range(rows):
            for x in range(cols):
                number = grid[y][x]
                if number < 0:
                    continue
                neigh = _neighbors(cols, rows, x, y)
                if any(grid[cy][cx] == UNKNOWN for cx, cy in neigh):
                    continue
                flagged = [cell for cell in neigh if cell in mines]
                unknown = [cell for cell in neigh if hidden_unknown(*cell)]
                need = number - len(flagged)
                if need < 0:
                    continue
                if need == 0 and unknown:
                    extra = set(unknown)
                    if not starves(extra):
                        safe |= extra
                        changed = True
                elif need == len(unknown) and unknown:
                    extra = set(unknown)
                    if not overflows(extra):
                        mines |= extra
                        changed = True
                elif unknown and need > 0:
                    constraints.append((frozenset(unknown), need))

        if changed:
            continue
        for index, (cells_a, need_a) in enumerate(constraints):
            for cells_b, need_b in constraints[index + 1 :]:
                if len(cells_a) > 8 or len(cells_b) > 8:
                    continue
                if cells_a < cells_b:
                    small, big = cells_a, cells_b
                    need_s, need_b2 = need_a, need_b
                elif cells_b < cells_a:
                    small, big = cells_b, cells_a
                    need_s, need_b2 = need_b, need_a
                else:
                    continue
                extra = big - small
                diff = need_b2 - need_s
                if diff == 0:
                    if not starves(set(extra)):
                        safe |= set(extra)
                        changed = True
                elif diff == len(extra):
                    if not overflows(set(extra)):
                        mines |= set(extra)
                        changed = True
                if changed:
                    break
            if changed:
                break

    # Cells that are mines or safe in every remaining configuration.
    probs = _probabilities(grid, mines, safe)
    forced_mines = {cell for cell, prob in probs.items() if prob >= 0.999}
    forced_safe = {cell for cell, prob in probs.items() if prob <= 0.001}
    if forced_mines and not overflows(forced_mines):
        mines |= forced_mines
    if forced_safe and not starves(forced_safe):
        safe |= forced_safe

    already_flagged = {(x, y) for y in range(rows) for x in range(cols) if grid[y][x] == FLAG}
    plan = Plan()
    plan.flags = sorted(mines - already_flagged)

    safe_hidden = sorted(cell for cell in safe if grid[cell[1]][cell[0]] == HIDDEN)
    covered: set[tuple[int, int]] = set()
    chord_candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for y in range(rows):
        for x in range(cols):
            number = grid[y][x]
            if number <= 0:
                continue
            neigh = _neighbors(cols, rows, x, y)
            flagged = sum(1 for cell in neigh if cell in mines or cell in already_flagged)
            opening = {cell for cell in neigh if cell in safe and grid[cell[1]][cell[0]] == HIDDEN}
            if flagged == number and opening:
                chord_candidates.append(((x, y), opening))
    # Greedy set cover so a cluster of numbers is not clicked five times.
    remaining = set(safe_hidden)
    while remaining and chord_candidates:
        chord_candidates.sort(key=lambda item: len(item[1] & remaining), reverse=True)
        cell, opening = chord_candidates.pop(0)
        hit = opening & remaining
        if not hit:
            continue
        plan.chords.append(cell)
        remaining -= hit
        covered |= hit
    plan.safe_cells = list(safe_hidden)
    plan.clicks = sorted(remaining)

    if plan.has_progress:
        return plan

    guess, risk = _best_guess(grid, mines, probs)
    plan.guess = guess
    plan.guess_risk = risk
    return plan


def _frontier(grid, mines, safe) -> tuple[list[tuple[int, int]], list[tuple[frozenset, int]]]:
    rows = len(grid)
    cols = len(grid[0])
    cells: set[tuple[int, int]] = set()
    constraints = []
    for y in range(rows):
        for x in range(cols):
            number = grid[y][x]
            if number < 0:
                continue
            neigh = _neighbors(cols, rows, x, y)
            if any(grid[cy][cx] == UNKNOWN for cx, cy in neigh):
                continue
            flagged = sum(1 for cell in neigh if cell in mines)
            unknown = [cell for cell in neigh if grid[cell[1]][cell[0]] == HIDDEN and cell not in mines and cell not in safe]
            need = number - flagged
            if unknown and 0 < need < len(unknown):
                constraints.append((frozenset(unknown), need))
                cells.update(unknown)
    return sorted(cells), constraints


def _components(cells, constraints):
    parent = {cell: cell for cell in cells}

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    linked = {cell: [] for cell in cells}
    for group, _need in constraints:
        group = [cell for cell in group if cell in parent]
        for other in group[1:]:
            union(group[0], other)
        for cell in group:
            linked[cell].append(group)
    groups: dict[tuple[int, int], list] = {}
    for cell in cells:
        groups.setdefault(find(cell), []).append(cell)
    out = []
    for members in groups.values():
        member_set = set(members)
        local = []
        seen = set()
        for cell in members:
            for group in linked[cell]:
                key = tuple(sorted(group))
                if key in seen:
                    continue
                if set(group) <= member_set:
                    seen.add(key)
                    need = next(need for cand, need in constraints if cand == frozenset(group))
                    local.append((frozenset(group), need))
        out.append((members, local))
    return out


def _enumerate(members, constraints) -> list[int] | None:
    """Bit masks of mine placements. None when the component is too big."""
    if len(members) > 16:
        return None
    index = {cell: bit for bit, cell in enumerate(members)}
    compiled = []
    for group, need in constraints:
        mask = 0
        for cell in group:
            mask |= 1 << index[cell]
        compiled.append((mask, need))
    solutions = []

    def popcount(value: int) -> int:
        return value.bit_count()

    def walk(bit: int, placed: int) -> None:
        if bit == len(members):
            solutions.append(placed)
            return
        for choice in (0, 1):
            nxt = placed | (choice << bit)
            ok = True
            for mask, need in compiled:
                decided = mask & ((1 << (bit + 1)) - 1)
                # Only prune once every cell in the constraint is decided,
                # plus the obvious over-mine check.
                mines_here = popcount(nxt & mask)
                if mines_here > need:
                    ok = False
                    break
                if decided == mask and mines_here != need:
                    ok = False
                    break
            if ok:
                walk(bit + 1, nxt)

    walk(0, 0)
    return solutions


def _probabilities(grid, mines, safe) -> dict[tuple[int, int], float]:
    cells, constraints = _frontier(grid, mines, safe)
    if not cells:
        return {}
    probs: dict[tuple[int, int], float] = {}
    for members, local in _components(cells, constraints):
        solutions = _enumerate(members, local)
        if not solutions:
            for cell in members:
                probs[cell] = 0.5
            continue
        for bit, cell in enumerate(members):
            mines_here = sum(1 for sol in solutions if sol & (1 << bit))
            probs[cell] = mines_here / len(solutions)
    return probs


def _best_guess(grid, mines, probs) -> tuple[tuple[int, int] | None, float]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    hidden = [(x, y) for y in range(rows) for x in range(cols) if grid[y][x] == HIDDEN and (x, y) not in mines]
    if not hidden:
        return None, 1.0
    if not probs:
        hidden.sort(key=lambda cell: abs(cell[0] - cols // 2) + abs(cell[1] - rows // 2))
        return hidden[0], 0.5

    def opening_potential(cell: tuple[int, int]) -> int:
        x, y = cell
        return sum(
            1
            for nx, ny in _neighbors(cols, rows, x, y)
            if grid[ny][nx] == HIDDEN and (nx, ny) not in mines
        )

    ranked = sorted(hidden, key=lambda cell: (probs.get(cell, 0.5), -opening_potential(cell)))
    choice = ranked[0]
    return choice, probs.get(choice, 0.5)
