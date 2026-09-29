"""Vicevrstva sit jako retezeni vrstev: vystup jedne vrstvy je vstupem dalsi.

`Sequential` nic nepocita sam - drzi seznam vrstev (kompozice, ne dedicnost)
a pri doprednem pruchodu je vola jednu po druhe. Vrstvou je zde `Neuron`
(modul `src.neuron`), jehoz `Linear` nese matici vah `(n_vstupu, n_jednotek)`
a vektor biasu `(n_jednotek,)`; vahy se do vrstvy vkladaji hotove zvenku,
navrzene rucne (zadne uceni - to prijde v Cviceni 10).

Behem pruchodu si sit uklada stopu `io_ = [x, out_1, out_2, ...]`. Mezilehle
vystupy (skryte vrstvy) jsou souradnice bodu v **novem prostoru priznaku**;
vizualizace (`dataio.plotting`) z nich kresli, jak skryta vrstva prostor
transformuje.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Sequential:
    """Retezec vrstev `f_L( ... f_2(f_1(x)) ... )` se zaznamem mezivystupu.

    Obdoba `nn.Sequential` v PyTorch. Vrstvy jsou injektovane zvenku
    (dependency injection) - `Sequential` si zadnou vrstvu nevytvari, jen je
    ulozi a ve spravnem poradi zavola. Kazda vrstva musi byt volatelna
    (`layer(x)`) a vracet 2D pole `(n_vzorku, n_jednotek)`, ktere je platnym
    vstupem nasledujici vrstvy.

    Atributy
    --------
    layers : list
        Vrstvy v poradi pruchodu (typicky instance `Neuron`).
    io_ : list[np.ndarray] | None
        Stopa posledniho doprednego pruchodu `[x, out_1, ..., out_L]`;
        pred prvnim volanim `forward` je `None`. Koncove podtrzitko znaci
        atribut vznikajici az behem vypoctu (konvence scikit-learn).
    """

    def __init__(self, layers: list) -> None:
        """Ulozi vrstvy site; stopa `io_` zatim neexistuje.

        Parametry
        ---------
        layers : list
            Seznam volatelnych vrstev (typicky `Neuron`) v poradi, v jakem jimi
            ma data projit.
        """
        self.layers: list[Any] = layers
        self.io_: list[np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Provede dopredny pruchod vsemi vrstvami a zaznamena mezivystupy.

        Definice
        --------
        io_[0]   = x
        io_[k]   = layers[k-1](io_[k-1])      pro k = 1, ..., L
        vystup   = io_[L]

        Vystup jedne vrstvy je vstupem nasledujici. Po skonceni pruchodu drzi
        `self.io_` seznam delky `1 + len(self.layers)`: vstup a vystupy vsech
        vrstev v poradi. `io_[1]` jsou tedy souradnice bodu v prostoru prvni
        skryte vrstvy - presne ty cte `plot_space_transformation`.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku tvaru `(n_vzorku, n_priznaku)`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vystup posledni vrstvy, tvaru `(n_vzorku, n_jednotek_posledni_vrstvy)`.
        """
        # assert  Ověřte, že seznam self.layers neni prazdny
        # assert  Ověřte, že x je 2D pole (n_vzorku, n_priznaku)
        raise NotImplementedError(
            "Úkol: nastavte self.io_ = [x]; pro kazdou vrstvu v self.layers "
            "spocitejte vystup z posledniho prvku self.io_ (out = layer(self.io_[-1])) "
            "a pripojte ho na konec self.io_; vratte self.io_[-1]."
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zkratka pro `forward` - umoznuje volat instanci jako funkci.

        Parametry
        ---------
        x : np.ndarray
            Matice vstupnich vzorku predana primo do `forward`.

        Navratova hodnota
        -----------------
        np.ndarray
            Vysledek `self.forward(x)`.
        """
        return self.forward(x)
