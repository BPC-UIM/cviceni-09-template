"""Generator nahodnych bodu v jednotkove hyperkrychli -- podklad pro plosne ulohy.

Na rozdil od hradel (ctyri rohy) pokryvaji tyto body cely ctverec
``[0, 1] x [0, 1]``, takze na nich jde videt, jak sit deli **plochu**
(trojuhelnik, motylek) a jak se tato plocha zobrazi do prostoru skryte vrstvy.
Stitky (ground truth) k bodum dodava modul ``dataio.shapes``.
"""

from __future__ import annotations

import numpy as np


def generate_points(
    n_samples: int, n_features: int = 2, seed: int | None = None
) -> np.ndarray:
    """Vygeneruje rovnomerne rozlozene nahodne body v ``[0, 1]^n_features``.

    Parametry
    ---------
    n_samples:
        Pocet bodu (``>= 1``).
    n_features:
        Rozmer prostoru; vychozi ``2`` (rovina ``x1-x2``).
    seed:
        Seed generatoru ``np.random.default_rng``. Stejny seed da stejne body
        (reprodukovatelnost); ``None`` da pri kazdem volani jine body.

    Navratova hodnota
    -----------------
    np.ndarray
        Matice tvaru ``(n_samples, n_features)`` typu ``float64``.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``n_samples < 1`` nebo ``n_features < 1``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples musi byt >= 1, zadano: {n_samples}")
    if n_features < 1:
        raise ValueError(f"n_features musi byt >= 1, zadano: {n_features}")

    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n_samples, n_features))
