"""Referencni oblasti (ground truth) v rovine: trojuhelnik a motylek.

Obe oblasti jsou definovany **geometricky**, nezavisle na jakekoli siti --
sit, kterou student navrhne, se s nimi porovnava.

Trojuhelnik
    Prunik tri polorovin ohranicenych primkami, na kterych lezi strany
    trojuhelniku. Konvexni oblast -> staci jedna skryta vrstva (tri jednotky,
    jedna na stranu) a vystupni jednotka typu AND.

Motylek
    Primky ``x2 = 0.5`` a ``x2 = x1`` se protinaji v bode ``(0.5, 0.5)`` a deli
    jednotkovy ctverec na ctyri klinove oblasti. Trida 1 jsou dva **protilehle
    (vrcholove) kliny** -- pravy horni a levy dolni; trida 0 zbytek. Presne
    ``(x2 > 0.5) != (x2 > x1)``, tedy XOR dvou polorovin. Oblast neni konvexni
    ani souvisla (kliny se dotykaji jen ve spolecnem vrcholu).
"""

from __future__ import annotations

import numpy as np


def inside_triangle(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Oznaci body lezici uvnitr trojuhelniku (vcetne hranice).

    Bod ``p`` lezi uvnitr, pokud je pro vsechny tri strany ``(a, b)`` na
    stejne strane primky jako protilehly vrchol. Test strany je znamenko
    vektoroveho soucinu ``(b - a) x (p - a)`` -- presne tatez polorovinova
    podminka, kterou realizuje jeden neuron skryte vrstvy. Vysledek nezavisi
    na poradi (orientaci) vrcholu.

    Parametry
    ---------
    points:
        Matice bodu tvaru ``(n, 2)``.
    vertices:
        Vrcholy trojuhelniku tvaru ``(3, 2)`` (seznam nebo ``np.ndarray``).

    Navratova hodnota
    -----------------
    np.ndarray
        Stitky tvaru ``(n,)`` typu ``int64``: ``1`` uvnitr, ``0`` venku.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``points`` nema tvar ``(n, 2)`` nebo ``vertices`` tvar ``(3, 2)``,
        nebo pokud je trojuhelnik degenerovany (nulovy obsah).
    """
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points musi mit tvar (n, 2), zadano: {points.shape}")
    if vertices.shape != (3, 2):
        raise ValueError(f"vertices musi mit tvar (3, 2), zadano: {vertices.shape}")

    def cross(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Z-slozka vektoroveho soucinu (b - a) x (p - a); p muze byt matice bodu."""
        return (b[0] - a[0]) * (p[..., 1] - a[1]) - (b[1] - a[1]) * (p[..., 0] - a[0])

    v0, v1, v2 = vertices
    orientation = cross(v0, v1, v2)  # znamenko = orientace vrcholu (CCW > 0)
    if np.isclose(orientation, 0.0):
        raise ValueError("Trojuhelnik je degenerovany (vrcholy lezi na jedne primce).")

    sign = np.sign(orientation)
    inside = np.ones(points.shape[0], dtype=bool)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        inside &= sign * cross(a, b, points) >= 0.0
    return inside.astype(np.int64)


def inside_butterfly(points: np.ndarray) -> np.ndarray:
    """Oznaci body lezici v "motylku" -- XOR dvou polorovin.

    Definice (pevna)
    ----------------
    stitek = (x2 > 0.5) != (x2 > x1)

    Trida 1 jsou dva protilehle kliny mezi primkami ``x2 = 0.5`` a ``x2 = x1``:
    pravy horni (``x2 > 0.5`` a zaroven ``x2 < x1``) a levy dolni
    (``x2 < 0.5`` a zaroven ``x2 > x1``). V jednotkovem ctverci zabira kazdy
    klin plochu 1/8, trida 1 tedy celkem 1/4 ctverce.

    Parametry
    ---------
    points:
        Matice bodu tvaru ``(n, 2)``; sloupce jsou ``x1``, ``x2``.

    Navratova hodnota
    -----------------
    np.ndarray
        Stitky tvaru ``(n,)`` typu ``int64``: ``1`` v motylku, ``0`` venku.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``points`` nema tvar ``(n, 2)``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points musi mit tvar (n, 2), zadano: {points.shape}")

    x1, x2 = points[:, 0], points[:, 1]
    return ((x2 > 0.5) != (x2 > x1)).astype(np.int64)
