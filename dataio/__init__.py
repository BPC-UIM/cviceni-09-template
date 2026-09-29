"""Verejne API balicku ``dataio`` pro cviceni 09.

Hradla (rohy jednotkoveho ctverce), nahodne body v rovine, referencni
oblasti (trojuhelnik, motylek), typovana konfigurace a vykreslovani site
vcetne transformace prostoru skrytou vrstvou.

Verejne API
-----------
- ``make_gate`` -- ``(x, y)`` pro hradlo z ``{"and", "or", "nand", "xor", "xnor"}``
- ``generate_points`` -- rovnomerne nahodne body v ``[0, 1]^n_features``
- ``inside_triangle`` -- stitky bodu uvnitr trojuhelniku (ground truth)
- ``inside_butterfly`` -- stitky bodu v motylku ``(x2 > 0.5) != (x2 > x1)``
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``DataConfig`` / ``SigmoidConfig`` / ``TriangleConfig`` / ``ExperimentConfig`` -- dataclassy
- ``NetworkVisualizer`` -- rozhodovaci oblast site, primky a aktivace skrytych jednotek
- ``plot_space_transformation`` -- body pred vrstvou a po ni (``io[0]`` vs. ``io[1]``)

Cely balicek ``dataio/`` je v tomto cviceni **predvyplneny** -- zadny
studentsky ukol, zadny ``NotImplementedError``.

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    ExperimentConfig,
    SigmoidConfig,
    TriangleConfig,
    load_config,
    validate_config,
)
from dataio.data import generate_points
from dataio.gates import make_gate
from dataio.plotting import NetworkVisualizer, plot_space_transformation
from dataio.shapes import inside_butterfly, inside_triangle

__all__ = [
    "make_gate",
    "generate_points",
    "inside_triangle",
    "inside_butterfly",
    "load_config",
    "validate_config",
    "DataConfig",
    "SigmoidConfig",
    "TriangleConfig",
    "ExperimentConfig",
    "NetworkVisualizer",
    "plot_space_transformation",
]
