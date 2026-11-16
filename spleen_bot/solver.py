"""Minesweeper deductions aimed at the fewest clicks.

Flags and chord clicks are preferred: one click on a satisfied number opens
every safe neighbor, which is faster than clicking those tiles one by one.
When the remaining-mine counter is known, mine counting and combinations drop
any placement that does not use exactly that many mines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

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


def _overflows(grid, mines: set[tuple[int, int]], extra: set[tuple[int, int]]) -> bool:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
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


def _starves(grid, mines: set[tuple[int, int]], safe: set[tuple[int, int]], extra_safe: set[tuple[int, int]]) -> bool:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
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


def _deduce_local(grid, mines: set[tuple[int, int]], safe: set[tuple[int, int]]) -> None:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def hidden_unknown(x: int, y: int) -> bool:
        return grid[y][x] == HIDDEN and (x, y) not in mines and (x, y) not in safe

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
                    if not _starves(grid, mines, safe, extra):
                        safe |= extra
                        changed = True
                elif need == len(unknown) and unknown:
                    extra = set(unknown)
                    if not _overflows(grid, mines, extra):
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
                    need_s, need_big = need_a, need_b
                elif cells_b < cells_a:
                    small, big = cells_b, cells_a
                    need_s, need_big = need_b, need_a
                else:
                    continue
                extra = big - small
                diff = need_big - need_s
                if diff == 0:
                    if not _starves(grid, mines, safe, set(extra)):
                        safe |= set(extra)
                        changed = True
                elif diff == len(extra):
                    if not _overflows(grid, mines, set(extra)):
                        mines |= set(extra)
                        changed = True
                if changed:
                    break
            if changed:
                break


def _force_probabilities(grid, mines: set[tuple[int, int]], safe: set[tuple[int, int]]) -> bool:
    """Flag or open cells that are mines or safe in every local solution."""
    probs = _probabilities(grid, mines, safe)
    forced_mines = {cell for cell, prob in probs.items() if prob >= 0.999}
    forced_safe = {cell for cell, prob in probs.items() if prob <= 0.001}
    accept_mines = bool(forced_mines) and not _overflows(grid, mines, forced_mines)
    accept_safe = bool(forced_safe) and not _starves(grid, mines, safe, forced_safe)
    if accept_mines:
        mines |= forced_mines
    if accept_safe:
        safe |= forced_safe
    return accept_mines or accept_safe


def _force_mine_count(grid, mines: set[tuple[int, int]], safe: set[tuple[int, int]], mines_left: int | None) -> bool:
    """Mine counting and combinations from the remaining-mine counter.

    The counter is how many mines are still unflagged. Local solutions that
    need a different number of mines are discarded. A cell that is a mine in
    every surviving solution is flagged; a cell that is safe in every one is
    opened. Separated hidden cells are safe when the constrained regions
    already use every remaining mine.
    """
    if mines_left is None:
        return False
    if any(value == UNKNOWN for row in grid for value in row):
        return False
    if _overflows(grid, mines, set()) or _starves(grid, mines, safe, set()):
        return False

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    unflagged = sum(1 for x, y in mines if grid[y][x] == HIDDEN)
    budget = mines_left - unflagged
    undecided = [
        (x, y)
        for y in range(rows)
        for x in range(cols)
        if grid[y][x] == HIDDEN and (x, y) not in mines and (x, y) not in safe
    ]
    if budget < 0 or budget > len(undecided) or not undecided:
        return False

    frontier, constraints = _frontier(grid, mines, safe)
    if budget == 0 and not frontier:
        if _starves(grid, mines, safe, set(undecided)):
            return False
        safe.update(undecided)
        return True
    if budget == len(undecided) and not frontier:
        if _overflows(grid, mines, set(undecided)):
            return False
        mines.update(undecided)
        return True

    regions: list[tuple[list[tuple[int, int]], list[int] | None]] = []
    seen: set[tuple[int, int]] = set()
    for members, local in _components(frontier, constraints):
        cap = None if len(members) <= 16 else 60_000
        solutions = _enumerate(members, local, max_mines=budget, node_budget=cap)
        # No placement fits the counter, or the region is too big to prove.
        if not solutions:
            return False
        regions.append((members, solutions))
        seen.update(members)

    interior = [cell for cell in undecided if cell not in seen]
    if interior:
        regions.append((interior, None))
    if not regions:
        return False

    options: list[list[int]] = []
    for members, solutions in regions:
        if solutions is None:
            options.append(list(range(min(len(members), budget) + 1)))
        else:
            options.append(sorted({sol.bit_count() for sol in solutions}))

    prefix = [{0}]
    for choices in options:
        nxt: set[int] = set()
        for so_far in prefix[-1]:
            for count in choices:
                if so_far + count <= budget:
                    nxt.add(so_far + count)
        prefix.append(nxt)
    if budget not in prefix[-1]:
        return False

    suffix: list[set[int]] = [set() for _ in regions]
    suffix.append({0})
    for index in range(len(regions) - 1, -1, -1):
        nxt = set()
        for so_far in suffix[index + 1]:
            for count in options[index]:
                if so_far + count <= budget:
                    nxt.add(so_far + count)
        suffix[index] = nxt

    new_mines: set[tuple[int, int]] = set()
    new_safe: set[tuple[int, int]] = set()
    for index, (members, solutions) in enumerate(regions):
        alive = [
            count
            for count in options[index]
            if any((budget - count - before) in suffix[index + 1] for before in prefix[index])
        ]
        if not alive:
            return False
        if solutions is None:
            if set(alive) == {0}:
                new_safe.update(members)
            elif set(alive) == {len(members)}:
                new_mines.update(members)
            continue
        full = (1 << len(members)) - 1
        mine_any: dict[int, int] = {}
        safe_any: dict[int, int] = {}
        for sol in solutions:
            count = sol.bit_count()
            mine_any[count] = mine_any.get(count, 0) | sol
            safe_any[count] = safe_any.get(count, 0) | (full ^ sol)
        for bit, cell in enumerate(members):
            mask = 1 << bit
            if all((safe_any.get(count, 0) & mask) == 0 for count in alive):
                new_mines.add(cell)
            elif all((mine_any.get(count, 0) & mask) == 0 for count in alive):
                new_safe.add(cell)

    new_mines -= mines
    new_safe -= safe
    if not new_mines and not new_safe:
        return False
    if new_mines and _overflows(grid, mines, new_mines):
        return False
    if new_safe and _starves(grid, mines, safe, new_safe):
        return False
    mines |= new_mines
    safe |= new_safe
    return True


def _count_probabilities(grid, mines, safe, mines_left: int | None) -> dict[tuple[int, int], float] | None:
    """Mine probability of each undecided cell under the remaining-mine counter.

    Placements that need more mines than the counter are dropped, and the
    rest are weighted by how many ways the other regions can make up the
    difference. None when the counter cannot be applied exactly.
    """
    if mines_left is None:
        return None
    if any(value == UNKNOWN for row in grid for value in row):
        return None
    if _overflows(grid, mines, set()) or _starves(grid, mines, safe, set()):
        return None
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    unflagged = sum(1 for x, y in mines if grid[y][x] == HIDDEN)
    budget = mines_left - unflagged
    undecided = [
        (x, y)
        for y in range(rows)
        for x in range(cols)
        if grid[y][x] == HIDDEN and (x, y) not in mines and (x, y) not in safe
    ]
    if budget < 0 or budget > len(undecided) or not undecided:
        return None
    frontier, constraints = _frontier(grid, mines, safe)
    if not frontier:
        if budget == 0 or budget == len(undecided):
            return None
        chance = budget / len(undecided)
        return {cell: chance for cell in undecided}

    regions: list[tuple[list[tuple[int, int]], list[int]]] = []
    seen: set[tuple[int, int]] = set()
    for members, local in _components(frontier, constraints):
        cap = None if len(members) <= 16 else 60_000
        solutions = _enumerate(members, local, max_mines=budget, node_budget=cap)
        if not solutions:
            return None
        regions.append((members, solutions))
        seen.update(members)
    interior = [cell for cell in undecided if cell not in seen]

    choices: list[list[tuple[int, float]]] = []
    for _members, solutions in regions:
        grouped: dict[int, int] = {}
        for sol in solutions:
            count = sol.bit_count()
            grouped[count] = grouped.get(count, 0) + 1
        choices.append([(count, float(amount)) for count, amount in sorted(grouped.items())])
    if interior:
        size = len(interior)
        choices.append([(count, float(comb(size, count))) for count in range(min(size, budget) + 1)])

    def convolve(base: list[float], options: list[tuple[int, float]]) -> list[float]:
        nxt = [0.0] * (budget + 1)
        for so_far, weight in enumerate(base):
            if weight == 0.0:
                continue
            for count, option_weight in options:
                total = so_far + count
                if total <= budget:
                    nxt[total] += weight * option_weight
        return nxt

    prefix = [[1.0] + [0.0] * budget]
    for options in choices:
        prefix.append(convolve(prefix[-1], options))
    if prefix[-1][budget] == 0.0:
        return None
    after = [[0.0] * (budget + 1) for _ in range(len(choices) + 1)]
    after[-1][0] = 1.0
    for index in range(len(choices) - 1, -1, -1):
        after[index] = convolve(after[index + 1], choices[index])
    total_weight = prefix[-1][budget]

    probs: dict[tuple[int, int], float] = {}
    for index, (members, solutions) in enumerate(regions):
        mine_weight = [0.0] * len(members)
        grouped_masks: dict[int, list[int]] = {}
        for sol in solutions:
            grouped_masks.setdefault(sol.bit_count(), []).append(sol)
        for count, group in grouped_masks.items():
            scale = 0.0
            for so_far, weight in enumerate(prefix[index]):
                rest = budget - count - so_far
                if weight and 0 <= rest <= budget:
                    scale += weight * after[index + 1][rest]
            if scale == 0.0:
                continue
            for sol in group:
                bit = 0
                while sol:
                    if sol & 1:
                        mine_weight[bit] += scale
                    sol >>= 1
                    bit += 1
        for bit, cell in enumerate(members):
            probs[cell] = min(1.0, max(0.0, mine_weight[bit] / total_weight))
    if interior:
        size = len(interior)
        expected = 0.0
        for count in range(min(size, budget) + 1):
            ways = 0.0
            for so_far, weight in enumerate(prefix[len(regions)]):
                rest = budget - count - so_far
                if weight and 0 <= rest <= budget:
                    ways += weight * after[len(regions) + 1][rest]
            expected += ways * comb(size, count) * count
        chance = min(1.0, max(0.0, (expected / size) / total_weight))
        for cell in interior:
            probs[cell] = chance
    return probs


def solve(grid: list[list[int]], mines_left: int | None = None) -> Plan:
    """Deduce flags and safe clicks.

    mines_left is the on-screen remaining-mine counter (total minus flags the
    game already shows). None keeps the solver on local logic only.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    mines: set[tuple[int, int]] = set()
    safe: set[tuple[int, int]] = set()
    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == FLAG:
                mines.add((x, y))

    _deduce_local(grid, mines, safe)
    for _ in range(4):
        grew = _force_probabilities(grid, mines, safe)
        grew = _force_mine_count(grid, mines, safe, mines_left) or grew
        if not grew:
            break
        _deduce_local(grid, mines, safe)

    already_flagged = {(x, y) for y in range(rows) for x in range(cols) if grid[y][x] == FLAG}
    plan = Plan()
    plan.flags = sorted(mines - already_flagged)
    new_flags = set(plan.flags)

    safe_hidden = sorted(cell for cell in safe if grid[cell[1]][cell[0]] == HIDDEN)
    covered: set[tuple[int, int]] = set()
    chord_candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for y in range(rows):
        for x in range(cols):
            number = grid[y][x]
            if number <= 0:
                continue
            neigh = _neighbors(cols, rows, x, y)
            # Chording opens every neighbor the game has not flagged yet. A mine
            # decided in this plan is still a plain hidden tile until that
            # right-click is accepted, so the chord would detonate it.
            if any(cell in new_flags for cell in neigh):
                continue
            flagged = sum(1 for cell in neigh if cell in already_flagged)
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

    probs = _count_probabilities(grid, mines, safe, mines_left)
    if probs is None:
        probs = _probabilities(grid, mines, safe)
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


def _enumerate(members, constraints, max_mines: int | None = None, node_budget: int | None = None) -> list[int] | None:
    """Bit masks of mine placements.

    None when the component is too big to finish. max_mines drops placements
    that already use more mines than the global counter can allow. A node
    budget keeps a large endgame search from stalling the click loop.
    """
    if not members:
        return []
    if node_budget is None and len(members) > 16:
        return None
    index = {cell: bit for bit, cell in enumerate(members)}
    compiled = []
    for group, need in constraints:
        mask = 0
        for cell in group:
            if cell not in index:
                continue
            mask |= 1 << index[cell]
        compiled.append((mask, need))
    solutions: list[int] = []
    nodes = 0
    aborted = False

    def walk(bit: int, placed: int) -> None:
        nonlocal nodes, aborted
        if aborted:
            return
        nodes += 1
        if node_budget is not None and nodes > node_budget:
            aborted = True
            return
        if bit == len(members):
            solutions.append(placed)
            return
        limit = (1 << (bit + 1)) - 1
        for choice in (0, 1):
            if aborted:
                return
            nxt = placed | (choice << bit)
            if max_mines is not None and nxt.bit_count() > max_mines:
                continue
            ok = True
            for mask, need in compiled:
                mines_here = (nxt & mask).bit_count()
                if mines_here > need:
                    ok = False
                    break
                undecided = (mask & ~limit).bit_count()
                if mines_here + undecided < need:
                    ok = False
                    break
                if undecided == 0 and mines_here != need:
                    ok = False
                    break
            if ok:
                walk(bit + 1, nxt)

    walk(0, 0)
    if aborted:
        return None
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

    ranked = sorted(
        hidden,
        key=lambda cell: (
            probs.get(cell, 0.5),
            -opening_potential(cell),
            abs(cell[0] - cols // 2) + abs(cell[1] - rows // 2),
        ),
    )
    choice = ranked[0]
    return choice, probs.get(choice, 0.5)
