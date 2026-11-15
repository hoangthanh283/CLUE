"""Bold the best cell of a table column; underline our best cell when a baseline is best.

Works on already formatted cells: ``$89.1{\\pm}2.0^{\\dagger}$``, ``$44.4/43.7$`` (per-seed
values, compared by their mean) and ``87.5\\;{\\scriptsize ...}``. Cells with no leading number
("--", "pending") are never marked. Ties (at the printed precision) are all bold.

    uv run python scripts/rca/table_emph.py   # self-check
"""

import re

_NUM = re.compile(r"\$?(\d+\.\d+(?:/\d+\.\d+)*)")
_MATH = re.compile(r"^\$(.+?)(\^\{.*\})?\$$")


def _value(cell: str) -> float | None:
    m = _NUM.match(cell)
    if not m:
        return None
    vs = [float(x) for x in m.group(1).split("/")]
    return round(sum(vs) / len(vs), 2)


def _wrap(cell: str, bold: bool) -> str:
    m = _MATH.match(cell)
    if m:
        cmd = "\\mathbf" if bold else "\\underline"
        return f"${cmd}{{{m.group(1)}}}{m.group(2) or ''}$"
    num = _NUM.match(cell).group(1)
    cmd = "\\textbf" if bold else "\\underline"
    return f"{cmd}{{{num}}}{cell[len(num):]}"


def emph(cells: list[str], ours: list[bool], skip: list[bool] | None = None) -> list[str]:
    """``cells`` of one column, top to bottom. ``skip`` rows (references, oracles) do not compete."""
    skip = skip or [False] * len(cells)
    vals = [None if s else _value(c) for c, s in zip(cells, skip, strict=True)]
    got = [v for v in vals if v is not None]
    if not got:
        return list(cells)
    best = max(got)
    out = [_wrap(c, True) if v == best else c for c, v in zip(cells, vals, strict=True)]
    mine = [v for v, o in zip(vals, ours, strict=True) if o and v is not None]
    if mine and max(mine) < best:
        out = [_wrap(c, False) if o and v == max(mine) else c
               for c, v, o in zip(out, vals, ours, strict=True)]  # fmt: skip
    return out


if __name__ == "__main__":
    col = ["$92.0{\\pm}0.4$", "$88.0{\\pm}0.9^{\\dagger}$", "$87.8{\\pm}0.5^{\\dagger}$", "--",
           "$86.0/88.0$"]  # fmt: skip
    got = emph(col, ours=[False, False, True, True, True], skip=[True, False, False, False, False])
    assert got == [col[0], "$\\mathbf{88.0{\\pm}0.9}^{\\dagger}$",
                   "$\\underline{87.8{\\pm}0.5}^{\\dagger}$", "--", col[4]], got  # fmt: skip
    assert emph(["1.0\\;{\\scriptsize $\\pm$ 0.1}", "2.0"], [True, True]) == [
        "1.0\\;{\\scriptsize $\\pm$ 0.1}",
        "\\textbf{2.0}",
    ]
    assert emph(["--", "--"], [True, False]) == ["--", "--"]
    print("ok")
