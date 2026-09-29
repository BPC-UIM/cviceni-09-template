"""Pravdivostni tabulky logickych hradel jako male 2D datasety (4 rohy ctverce).

Hradla slouzi v Cviceni 09 dvema ucelum: jako hotove referencni priklady
jednovrstve site (AND, OR -- linearne separovatelna) a jako nejmensi mozna
ukazka **transformace prostoru** (XOR -- linearne neseparovatelne v prostoru
``x1-x2``, ale separovatelne v prostoru skryte vrstvy ``y1-y2``).

======  ================  ==========================
hradlo  vystup ``y``       linearne separovatelne?
======  ================  ==========================
AND     ``[0, 0, 0, 1]``  ano
OR      ``[0, 1, 1, 1]``  ano
NAND    ``[1, 1, 1, 0]``  ano (negace AND)
XOR     ``[0, 1, 1, 0]``  **ne** (Cviceni 08) -- resi dvouvrstva sit
XNOR    ``[1, 0, 0, 1]``  **ne** (negace XOR)
======  ================  ==========================

Poradi radku ``x`` je ve vsech hradlech stejne: ``[0,0], [0,1], [1,0], [1,1]``.
"""

from __future__ import annotations

import numpy as np

# Ctyri kombinace vstupu v pevnem poradi; sdilene vsemi hradly.
_INPUTS: np.ndarray = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
)

# Pravdivostni tabulky (vystup pro poradi radku v ``_INPUTS``).
_TRUTH_TABLES: dict[str, list[int]] = {
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "nand": [1, 1, 1, 0],
    "xor": [0, 1, 1, 0],
    "xnor": [1, 0, 0, 1],
}


def make_gate(gate: str) -> tuple[np.ndarray, np.ndarray]:
    """Vrati ``(x, y)`` pro zadane logicke hradlo.

    Parametry
    ---------
    gate:
        Jmeno hradla z mnoziny ``{"and", "or", "nand", "xor", "xnor"}``.
        Velikost pismen ani okrajove mezery nevadi -- hodnota se normalizuje
        pres ``gate.lower().strip()``.

    Navratova hodnota
    -----------------
    x:
        ``np.ndarray`` tvaru ``(4, 2)`` typu ``float64`` se ctyrmi radky
        ``[0, 0]``, ``[0, 1]``, ``[1, 0]``, ``[1, 1]`` v tomto poradi.
    y:
        ``np.ndarray`` tvaru ``(4,)`` typu ``int64`` s vystupem hradla pro
        odpovidajici radky ``x``.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``gate`` po normalizaci neni jednim z podporovanych hradel.
    """
    key = gate.lower().strip()
    if key not in _TRUTH_TABLES:
        povolene = ", ".join(sorted(_TRUTH_TABLES))
        raise ValueError(
            f"Nezname hradlo: {gate!r}. Povolene hodnoty jsou: {povolene}."
        )

    x = _INPUTS.copy()
    y = np.array(_TRUTH_TABLES[key], dtype=np.int64)
    return x, y
