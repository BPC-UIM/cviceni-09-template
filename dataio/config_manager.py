"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 09.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a dve
funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje a
vrati) a ``validate_config`` (rozsahove kontroly s ceskymi chybovymi
hlaskami). Cely modul je predvyplneny -- cviceni neobsahuje vyber
implementace retezcem (Strategy + Factory), konfigurace je ciste
infrastruktura.

K hodnotam se pristupuje pres atributy (napr. ``cfg.sigmoid.temperature``),
nikdy ne pres klice slovniku -- preklep v atributu odhali editor/typovy
kontroler staticky, zatimco ``cfg["sigmoid"]["temperature"]`` spadne az za
behu.

Vahy a biasy siti v konfiguraci **nejsou** -- navrhuje je student primo ve
fazich ``cviceni_09.py``, protoze jejich tvar (matice) je soucasti navrhu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Nastaveni nahodnych bodu (sekce ``data``)."""

    n_samples: int
    seed: int | None


@dataclass
class SigmoidConfig:
    """Nastaveni sigmoidove aktivace ve vrstvach site (sekce ``sigmoid``)."""

    temperature: float


@dataclass
class TriangleConfig:
    """Geometrie trojuhelniku (sekce ``triangle``).

    Atributy
    --------
    vertices:
        Seznam tri vrcholu ``[[x1, x2], [x1, x2], [x1, x2]]``.
    """

    vertices: list[list[float]]


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    data: DataConfig
    sigmoid: SigmoidConfig
    triangle: TriangleConfig


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``DataConfig``,
    ``SigmoidConfig`` a ``TriangleConfig``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud chybi sekce/klic, nebo hodnota nesplnuje kontroly
        ve ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    try:
        seed_raw = raw["data"]["seed"]
        cfg = ExperimentConfig(
            data=DataConfig(
                n_samples=int(raw["data"]["n_samples"]),
                seed=int(seed_raw) if seed_raw is not None else None,
            ),
            sigmoid=SigmoidConfig(
                temperature=float(raw["sigmoid"]["temperature"]),
            ),
            triangle=TriangleConfig(
                vertices=[[float(c) for c in vertex] for vertex in raw["triangle"]["vertices"]],
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"config.yaml nema ocekavanou strukturu (chybi {exc})") from exc

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou obsahujici zadanou hodnotu. Kontroluji se:

    - ``data.n_samples >= 1``
    - ``sigmoid.temperature > 0``
    - ``triangle.vertices`` ma tvar 3 x 2 (tri vrcholy, kazdy se dvema
      souradnicemi)

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    if cfg.data.n_samples < 1:
        raise ValueError(f"data.n_samples musi byt >= 1, zadano: {cfg.data.n_samples}")
    if cfg.sigmoid.temperature <= 0:
        raise ValueError(
            f"sigmoid.temperature musi byt > 0, zadano: {cfg.sigmoid.temperature}"
        )

    vertices = cfg.triangle.vertices
    if len(vertices) != 3 or any(len(vertex) != 2 for vertex in vertices):
        raise ValueError(
            "triangle.vertices musi byt 3 vrcholy po 2 souradnicich (tvar 3 x 2), "
            f"zadano: {vertices}"
        )
